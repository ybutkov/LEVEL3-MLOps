"""Resource for publishing versioned dataset snapshots to LakeFS.

Write-side counterpart of :class:`LakeFSSourceResource` (which reads raw data):
``publish`` puts a set of files on an isolated per-run branch, commits them, and
merges into ``main`` — so a run's whole dataset snapshot lands atomically and the
returned commit id is an immutable handle to "the data this run produced".
"""

import dagster as dg
import lakefs
from lakefs.client import Client
from lakefs.exceptions import BadRequestException

class LakeFSVersioningResource(dg.ConfigurableResource):
    """Commit a batch of files to LakeFS as one versioned snapshot.

    Parameters
    ----------
    host : str
        LakeFS server endpoint.
    repo : str
        Repository name.
    access_key, secret_key : str
        LakeFS credentials (provided via env).
    """

    host: str
    repo: str
    access_key: str
    secret_key: str

    def publish(
        self,
        *,
        files: dict[str, bytes],
        branch: str,
        message: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload ``files`` on ``branch`` (off main), commit, merge into main.

        Parameters
        ----------
        files : dict[str, bytes]
            Mapping of in-repo path -> file content.
        branch : str
            Isolated working branch (created off ``main`` if absent).
        message : str
            Commit message.
        metadata : dict[str, str] | None, default=None
            Commit metadata (e.g. run id, code revision).

        Returns
        -------
        str
            The snapshot's commit id (on ``main`` after the merge).
        """

        client = Client(host=self.host, username=self.access_key, password=self.secret_key)
        repo = lakefs.Repository(self.repo, client=client)

        work = repo.branch(branch).create(source_reference="main", exist_ok=True)
        for path, blob in files.items():
            work.object(path).upload(blob, pre_sign=False)
        try:
            commit_id = work.commit(message=message, metadata=metadata or {}).id
            work.merge_into("main")
        except BadRequestException as error:
            # identical to main (no diff): the data is already published there,
            # so the current head is the right version to return.
            if "no changes" not in str(error).lower():
                raise
            commit_id = work.head.id
        return commit_id

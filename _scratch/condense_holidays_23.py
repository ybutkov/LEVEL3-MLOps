"""Condense §2 (temperature) + §3 (growth) of exploration_holidays.ipynb into one short section.
Keeps `residual`/`reg_std` (used by §4); drops both plots and the long narrative."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NB = "notebooks/exploration_holidays.ipynb"
nb = nbformat.read(NB, as_version=4)

OLD = ["q2-hdr", "q2-code", "q2-plot", "q2-obs", "q3-hdr", "q3-within-holiday", "q3-obs"]
ids = [c.get("id") for c in nb.cells]
i0, i1 = ids.index("q2-hdr"), ids.index("q3-obs")

hdr = new_markdown_cell("""## 2. Перекрёстные проверки (кратко)

---

Локальная база ±15 дней из §1 уже контролирует и рост, и сезон. Ниже — компактная запись двух проверок,
которые к ней привели (раньше это были §2 «температура» и §3 «рост» с полными графиками):
объясняет ли разброс **температура** и **рост сервиса**.

`residual` (факт − линейный тренд по температуре) считаем здесь — он используется ниже в §4.""")
hdr["id"] = "q2-checks-hdr"

code = new_code_cell("""\
# Контроль на температуру: линейный тренд по обычным дням, остаток = факт − предсказание.
coef = np.polyfit(regular["avg_temp"], regular["daily_rentals"], 1)
pred_fn = np.poly1d(coef)

regular = regular.copy()
holidays_df = holidays_df.copy()
regular["residual"] = regular["daily_rentals"] - pred_fn(regular["avg_temp"])
holidays_df["residual"] = holidays_df["daily_rentals"] - pred_fn(holidays_df["avg_temp"])

reg_std = regular["residual"].std()
outside_1std = (holidays_df["residual"].abs() > reg_std).sum()

# Рост: вариация одного и того же праздника между годами (в остатках).
paired = holidays_df.pivot_table(index="holiday", columns="year", values="residual").dropna()
cross_year = (paired[2012] - paired[2011]).abs().mean()

print(f"std остатков обычных дней:       {reg_std:,.0f}")
print(f"праздников за ±1 std после t°:    {outside_1std} / {len(holidays_df)}  -> температура не объясняет")
print(f"среднее |разницы| год-к-году:    {cross_year:,.0f}  (~ std обычных дней)\")""")
code["id"] = "q2-checks-code"

obs = new_markdown_cell("""### Наблюдения

- **Температура не объясняет разброс:** даже после контроля на t° **8 / 21** праздников выходят за ±1 std обычных дней (std ≈ 1,500). Дело не в погоде.
- **Рост — главный confound:** межгодовая вариация одного праздника в остатках (~1,660) сопоставима со std обычных дней (~1,500), поэтому раньше выглядела как шум. Но §1 (локальная % база) показал: это был **именно рост** — убрали его, и праздник год-к-году устойчив.
- Обе проверки **поглощены локальной базой §1**; оставлены компактно как «как мы сюда пришли». `residual` ниже используется в §4.""")
obs["id"] = "q2-checks-obs"

nb.cells[i0:i1 + 1] = [hdr, code, obs]

# keep Выводы coherent: §3 больше нет, и §1 смягчил тезис про «группировку»
for c in nb.cells:
    if c.get("id") == "conclusions-hdr":
        s = c["source"]
        s = s.replace(
            "- Группировка/типизация праздников бесполезна: слишком мало данных, вариация внутри группы ≈ шум (§3).",
            "- Разброс праздников — это рост + сезон, а не «тип»; локальная база ±15 дней (§1) их убирает. После контроля праздник **устойчив год-к-году** → типизация *могла бы* нести сигнал, но на 21 дне ненадёжна — решает **ablation на тесте**, а не наш глаз.",
        )
        s = s.replace("Температура не объясняет разброс (§2).", "Температура не объясняет разброс (§2).")
        c["source"] = s

nbformat.write(nb, NB)
print(f"replaced cells [{i0}:{i1}] ({len(OLD)} -> 3); total now {len(nb.cells)}")

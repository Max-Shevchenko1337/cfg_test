import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from datetime import datetime

# ==========================================
# 1. ВХІДНІ ДАНІ
# ==========================================

# Врожайність полів по днях (тони)
HARVEST_DATA = {
    'F1': [200, 400, 600, 200, 0, 0, 0, 0, 0, 0],
    'F2': [200, 400, 600, 200, 0, 0, 0, 0, 0, 0],
    'F3': [200, 400, 500, 200, 0, 0, 0, 0, 0, 0],
    'F4': [0, 200, 400, 400, 200, 0, 0, 0, 0, 0],
    'F5': [0, 0, 400, 600, 400, 0, 0, 0, 0, 0],
    'F6': [0, 0, 0, 600, 600, 400, 0, 0, 0, 0],
    'F7': [0, 0, 0, 0, 500, 500, 200, 0, 0, 0],
    'F8': [0, 0, 0, 0, 500, 500, 400, 0, 0, 0],
    'F9': [0, 0, 0, 0, 500, 400, 300, 0, 0, 0],
    'F10': [0, 0, 0, 0, 500, 300, 200, 0, 0, 0]
}

# Вартість логістики ($/т) від поля до елеватора
COSTS_DATA = {
    'F1': {'Elev_A': 8, 'Elev_B': 12, 'Elev_C': 20},
    'F2': {'Elev_A': 9, 'Elev_B': 11, 'Elev_C': 19},
    'F3': {'Elev_A': 10, 'Elev_B': 10, 'Elev_C': 18},
    'F4': {'Elev_A': 14, 'Elev_B': 9, 'Elev_C': 16},
    'F5': {'Elev_A': 18, 'Elev_B': 8, 'Elev_C': 14},
    'F6': {'Elev_A': 20, 'Elev_B': 14, 'Elev_C': 9},
    'F7': {'Elev_A': 25, 'Elev_B': 16, 'Elev_C': 8},
    'F8': {'Elev_A': 30, 'Elev_B': 18, 'Elev_C': 10},
    'F9': {'Elev_A': 32, 'Elev_B': 20, 'Elev_C': 11},
    'F10': {'Elev_A': 35, 'Elev_B': 22, 'Elev_C': 12}
}

# Характеристики елеваторів (Cap - місткість, In_Lim - ліміт приймання, Out_Lim - ліміт відвантаження)
ELEVATORS_SPECS = {
    'Elev_A': {'Cap': 1200, 'In_Lim': 800, 'Out_Lim': 500},
    'Elev_B': {'Cap': 4500, 'In_Lim': 1500, 'Out_Lim': 1200},
    'Elev_C': {'Cap': 3000, 'In_Lim': 1500, 'Out_Lim': 1200}
}

# План продажів (відвантаження) по днях
SALES_TARGETS = [0, 0, 500, 1500, 2000, 2500, 2000, 1500, 1000, 500]

# Поріг завантаження для визначення "вузького місця" (%)
BOTTLENECK_THRESHOLD = 90


# ==========================================
# 2. ФУНКЦІЇ ЛОГІКИ ТА ЗВІТІВ
# ==========================================

def validate_input_data() -> bool:
    """Перевіряє коректність вхідних даних перед запуском."""
    errors = []
    if len(SALES_TARGETS) != 10:
        errors.append("План продажів має бути розрахований на 10 днів.")

    total_harvest = sum(sum(h) for h in HARVEST_DATA.values())
    total_sales = sum(SALES_TARGETS)

    if total_harvest < total_sales:
        errors.append(f"Загальний врожай ({total_harvest}) менший за план продажів ({total_sales}).")

    if errors:
        print("\n!!! ЗНАЙДЕНО ПОМИЛКИ У ДАНИХ !!!")
        for err in errors: print(f"- {err}")
        return False
    return True


def create_optimization_model():
    """Створює математичну модель оптимізації за допомогою бібліотеки PuLP."""
    Fields = list(HARVEST_DATA.keys())
    Elevators = list(ELEVATORS_SPECS.keys())
    Days = range(10)

    # Ініціалізація проблеми (Мінімізація витрат)
    prob = pulp.LpProblem("AgroLogistics_UA", pulp.LpMinimize)

    # Змінні рішення
    x = pulp.LpVariable.dicts("Flow", (Fields, Elevators, Days), lowBound=0, cat='Continuous')  # Потік
    out = pulp.LpVariable.dicts("Out", (Elevators, Days), lowBound=0, cat='Continuous')  # Відвантаження
    stock = pulp.LpVariable.dicts("Stock", (Elevators, Days), lowBound=0, cat='Continuous')  # Залишки

    # Цільова функція: Мінімізація вартості перевезень
    prob += pulp.lpSum(x[f][e][t] * COSTS_DATA[f][e] for f in Fields for e in Elevators for t in Days)

    # Обмеження
    for t in Days:
        # 1. Весь врожай має бути вивезений з поля в день збору
        for f in Fields:
            prob += pulp.lpSum(x[f][e][t] for e in Elevators) == HARVEST_DATA[f][t]

        # 2. Виконання плану продажів (сумарно по всіх елеваторах)
        prob += pulp.lpSum(out[e][t] for e in Elevators) == SALES_TARGETS[t]

        for e in Elevators:
            inbound = pulp.lpSum(x[f][e][t] for f in Fields)

            # 3. Ліміт на приймання елеватора
            prob += inbound <= ELEVATORS_SPECS[e]['In_Lim']

            # 4. Ліміт на відвантаження елеватора
            prob += out[e][t] <= ELEVATORS_SPECS[e]['Out_Lim']

            # 5. Баланс запасів (Залишок = Попередній + Прихід - Відхід)
            if t == 0:
                prob += stock[e][t] == inbound - out[e][t]
            else:
                prob += stock[e][t] == stock[e][t - 1] + inbound - out[e][t]

            # 6. Ліміт місткості зберігання
            prob += stock[e][t] <= ELEVATORS_SPECS[e]['Cap']

    return prob, x, out, stock


def analyze_bottlenecks(x, out, stock):
    """Аналізує завантаженість системи та шукає критичні точки."""
    bottlenecks = []
    Fields = list(HARVEST_DATA.keys())
    Elevators = list(ELEVATORS_SPECS.keys())
    Days = range(10)

    for t in Days:
        for e in Elevators:
            # Перевірка входу (Приймання)
            inbound = sum(x[f][e][t].varValue or 0 for f in Fields)
            if ELEVATORS_SPECS[e]['In_Lim'] > 0:
                pct = (inbound / ELEVATORS_SPECS[e]['In_Lim'] * 100)
                if pct >= BOTTLENECK_THRESHOLD:
                    bottlenecks.append({
                        'День': t + 1, 'Елеватор': e, 'Тип': 'Приймання',
                        'Завантаження %': round(pct, 1), 'Факт': round(inbound, 1),
                        'Ліміт': ELEVATORS_SPECS[e]['In_Lim']
                    })

            # Перевірка виходу (Відвантаження)
            outbound = out[e][t].varValue or 0
            if ELEVATORS_SPECS[e]['Out_Lim'] > 0:
                pct = (outbound / ELEVATORS_SPECS[e]['Out_Lim'] * 100)
                if pct >= BOTTLENECK_THRESHOLD:
                    bottlenecks.append({
                        'День': t + 1, 'Елеватор': e, 'Тип': 'Відвантаження',
                        'Завантаження %': round(pct, 1), 'Факт': round(outbound, 1),
                        'Ліміт': ELEVATORS_SPECS[e]['Out_Lim']
                    })

            # Перевірка залишків (Місткість)
            st = stock[e][t].varValue or 0
            if ELEVATORS_SPECS[e]['Cap'] > 0:
                pct = (st / ELEVATORS_SPECS[e]['Cap'] * 100)
                if pct >= BOTTLENECK_THRESHOLD:
                    bottlenecks.append({
                        'День': t + 1, 'Елеватор': e, 'Тип': 'Місткість',
                        'Завантаження %': round(pct, 1), 'Факт': round(st, 1),
                        'Ліміт': ELEVATORS_SPECS[e]['Cap']
                    })
    return bottlenecks


def create_flow_table(x):
    """Створює детальний звіт по переміщенню зерна."""
    flows = []
    for t in range(10):
        for f in HARVEST_DATA:
            for e in ELEVATORS_SPECS:
                val = x[f][e][t].varValue
                if val and val > 0.01:
                    flows.append({
                        'День': t + 1, 'Поле': f, 'Елеватор': e,
                        'Тони': round(val, 2), 'Тариф ($)': COSTS_DATA[f][e],
                        'Сума ($)': round(val * COSTS_DATA[f][e], 2)
                    })
    return pd.DataFrame(flows)


def create_elevator_summary(stock, out, x):
    """Звіт про роботу елеваторів по днях."""
    summary = []
    for t in range(10):
        for e in ELEVATORS_SPECS:
            inbound = sum(x[f][e][t].varValue or 0 for f in HARVEST_DATA)
            summary.append({
                'День': t + 1, 'Елеватор': e,
                'Прийнято': round(inbound, 2), 'Ліміт вх.': ELEVATORS_SPECS[e]['In_Lim'],
                'Відвантажено': round(out[e][t].varValue or 0, 2), 'Ліміт вих.': ELEVATORS_SPECS[e]['Out_Lim'],
                'Залишок': round(stock[e][t].varValue or 0, 2), 'Місткість': ELEVATORS_SPECS[e]['Cap']
            })
    return pd.DataFrame(summary)


def create_daily_cost_summary(x):
    """Звіт про щоденні витрати."""
    costs = []
    for t in range(10):
        d_cost = sum((x[f][e][t].varValue or 0) * COSTS_DATA[f][e] for f in HARVEST_DATA for e in ELEVATORS_SPECS)
        d_vol = sum((x[f][e][t].varValue or 0) for f in HARVEST_DATA for e in ELEVATORS_SPECS)
        costs.append({
            'День': t + 1, 'Обсяг (т)': round(d_vol, 2), 'Витрати ($)': round(d_cost, 2),
            'Середня ціна ($/т)': round(d_cost / d_vol, 2) if d_vol else 0
        })
    return pd.DataFrame(costs)


def create_route_analysis(x):
    """Аналіз використаних маршрутів."""
    routes = []
    for f in HARVEST_DATA:
        for e in ELEVATORS_SPECS:
            vol = sum((x[f][e][t].varValue or 0) for t in range(10))
            if vol > 0.01:
                routes.append({
                    'Поле': f, 'Елеватор': e, 'Всього (т)': round(vol, 2),
                    'Тариф ($)': COSTS_DATA[f][e],
                    'Сума ($)': round(vol * COSTS_DATA[f][e], 2)
                })
    df = pd.DataFrame(routes)
    if not df.empty:
        df = df.sort_values('Всього (т)', ascending=False)
    return df


def create_elevator_efficiency(stock, out, x):
    """Зведена ефективність елеваторів."""
    eff = []
    for e in ELEVATORS_SPECS:
        total_in = sum(sum(x[f][e][t].varValue or 0 for f in HARVEST_DATA) for t in range(10))
        total_out = sum(out[e][t].varValue or 0 for t in range(10))
        avg_stock = sum(stock[e][t].varValue or 0 for t in range(10)) / 10
        eff.append({
            'Елеватор': e, 'Всього прийнято': round(total_in, 2),
            'Всього відвантажено': round(total_out, 2),
            'Середній залишок': round(avg_stock, 2)
        })
    return pd.DataFrame(eff)


def create_daily_balance(x, out, stock):
    """Загальний баланс системи."""
    bal = []
    for t in range(10):
        inb = sum(sum(x[f][e][t].varValue or 0 for f in HARVEST_DATA) for e in ELEVATORS_SPECS)
        outb = sum(out[e][t].varValue or 0 for e in ELEVATORS_SPECS)
        st = sum(stock[e][t].varValue or 0 for e in ELEVATORS_SPECS)
        bal.append({
            'День': t + 1, 'Збір врожаю': sum(HARVEST_DATA[f][t] for f in HARVEST_DATA),
            'Прийнято системою': round(inb, 2),
            'План продажів': SALES_TARGETS[t],
            'Відвантажено факт': round(outb, 2), 'На складах': round(st, 2)
        })
    return pd.DataFrame(bal)


def create_answers_summary(prob, x, out, stock, bottlenecks):
    """Формування підсумкової сторінки."""
    total_cost = pulp.value(prob.objective)
    status_map = {1: 'Оптимально', 0: 'Не вирішено', -1: 'Неможливо', -2: 'Необмежено', -3: 'Невизначено'}
    status_text = status_map.get(prob.status, 'Невідомий статус')

    data = [
        {'Розділ': 'РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ', 'Інформація': ''},
        {'Розділ': '1. Загальна вартість логістики', 'Інформація': f"${total_cost:,.2f}"},
        {'Розділ': '2. Статус рішення', 'Інформація': status_text},
        {'Розділ': '3. Вузькі місця (зав.>90%)',
         'Інформація': f"{len(bottlenecks)} критичних подій" if bottlenecks else "Система працює без перевантажень"},
    ]

    if bottlenecks:
        data.append({'Розділ': '--- ДЕТАЛІ КРИТИЧНИХ ТОЧОК ---', 'Інформація': ''})
        for b in bottlenecks:
            info_str = f"День {b['День']} | {b['Елеватор']} | {b['Тип']} | {b['Завантаження %']}% ({b['Факт']}/{b['Ліміт']})"
            data.append({'Розділ': 'Критично', 'Інформація': info_str})

    return pd.DataFrame(data)


# ==========================================
# 3. ГОЛОВНА ФУНКЦІЯ
# ==========================================

def main():
    print("--- ЗАПУСК ОПТИМІЗАЦІЇ АГРОЛОГІСТИКИ ---")

    # 1. Валідація
    if not validate_input_data():
        return

    # 2. Розрахунок
    print("-> Створення та розв'язання моделі...", end=" ")
    prob, x, out, stock = create_optimization_model()

    # Вимкнення повідомлень солвера (msg=0) для чистоти консолі
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status_code = prob.status
    print(f"[{'ОК' if status_code == 1 else 'Помилка'}]")

    if status_code != 1:
        print("Помилка: Оптимальне рішення не знайдено. Перевірте обмеження.")
        return

    # 3. Вивід основних результатів у консоль
    total_cost = pulp.value(prob.objective)
    bottlenecks = analyze_bottlenecks(x, out, stock)

    print(f"-> Мінімальна вартість: ${total_cost:,.2f}")
    print(f"-> Знайдено вузьких місць: {len(bottlenecks)}")

    # 4. Генерація повного Excel звіту
    print("-> Формування Excel файлу...", end=" ")

    try:
        output_dir = 'excel_reports'
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        filename = f"agro_logistics_ua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        full_path = os.path.join(output_dir, filename)

        # Створення всіх таблиць
        answers_df = create_answers_summary(prob, x, out, stock, bottlenecks)
        flow_df = create_flow_table(x)
        route_df = create_route_analysis(x)
        balance_df = create_daily_balance(x, out, stock)
        elevator_df = create_elevator_summary(stock, out, x)
        efficiency_df = create_elevator_efficiency(stock, out, x)
        cost_df = create_daily_cost_summary(x)

        if bottlenecks:
            bottleneck_df = pd.DataFrame(bottlenecks).sort_values('Завантаження %', ascending=False)
        else:
            bottleneck_df = None

        # Запис у файл
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            answers_df.to_excel(writer, sheet_name='Головна', index=False)
            flow_df.to_excel(writer, sheet_name='Маршрути (детально)', index=False)
            route_df.to_excel(writer, sheet_name='Аналіз напрямків', index=False)
            balance_df.to_excel(writer, sheet_name='Баланс системи', index=False)
            elevator_df.to_excel(writer, sheet_name='Стан елеваторів', index=False)
            efficiency_df.to_excel(writer, sheet_name='КПЕ елеваторів', index=False)
            cost_df.to_excel(writer, sheet_name='Витрати по днях', index=False)

            if bottleneck_df is not None:
                bottleneck_df.to_excel(writer, sheet_name='Вузькі місця', index=False)

            # Автоматичне налаштування ширини колонок для краси
            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    sheet.column_dimensions[col[0].column_letter].width = 20

        print("Готово.")
        print(f"-> Файл успішно збережено: {os.path.abspath(full_path)}")

    except Exception as e:
        print(f"\nПомилка при збереженні файлу: {e}")
        print("Переконайтеся, що файл не відкритий в Excel.")


if __name__ == "__main__":
    main()
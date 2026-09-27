"""Small bilingual presentation layer for the Streamlit product.

English phrases are stable translation keys. Business identifiers, export
schemas, channel names, and mathematical notation remain language-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


APP_NAME = "MediaPlan Optimizer"
APP_SLUG = "mediaplan_optimizer"

LANGUAGES = {"ru": "Русский", "en": "English"}

RU = {
    "Demo calculated: Calendar-aware, 21 days, RUB 1,200,000, maximum conversions, seed 42.": "Демо рассчитано: Calendar-aware, 21 день, 1 200 000 ₽, максимум конверсий, seed 42.",
    "Start campaign workflow": "Начать ведение кампании",
    "The campaign is tracked inside the prototype; no external advertising platforms or APIs are called.": "Кампания ведётся внутри прототипа: внешние рекламные кабинеты и API не вызываются.",
    "The simulator does not generate observed reach. Upload observed campaign data via CSV with a reach value.": "Для кампании с целью по охвату синтетический симулятор недоступен: он не генерирует наблюдаемый reach. Загрузите фактические данные через CSV с полем reach.",
    "Online bandit policies support clicks and conversions only, not reach.": "Онлайн-бандиты поддерживают только клики и конверсии, но не охват.",
    "Hidden simulator parameters may differ by up to approximately ±30% from planning assumptions.": "При значении 0,30 скрытые параметры симулятора могут отклоняться от плановых примерно на ±30%.",
    "Static baseline (Uniform V1)": "Static baseline (Uniform V1)",
    "The research comparison uses a common Uniform V1 baseline and a synthetic environment. It is separate from the main Calendar-aware media planner.": "Исследовательское сравнение использует единый Uniform V1 baseline и синтетическую среду. Оно отделено от основного Calendar-aware медиапланировщика.",
    "Thompson Sampling and LinUCB are research policies evaluated in a synthetic environment, not production advertising-platform controllers.": "Thompson Sampling и LinUCB — исследовательские стратегии для синтетической среды, а не готовые системы управления рекламными платформами.",
    "LinUCB is an experimental contextual strategy. Results depend on exploration alpha and the synthetic scenario.": "LinUCB — экспериментальная контекстная стратегия. Результат зависит от коэффициента исследования alpha и синтетического сценария.",
    "Compare achieved KPI for the same budget, channels, horizon, and catalog.": "Сравнение достигнутого KPI при одинаковом бюджете, наборе каналов, сроке и каталоге.",
    "Uniform baseline is solved independently for the same target, channels, horizon, and catalog. Compare required model budgets.": "Uniform baseline рассчитывается независимо для той же цели, каналов, срока и каталога. Сравниваются необходимые модельные бюджеты.",
    "Calculate baseline": "Рассчитать baseline",
    "Calculating Uniform baseline…": "Рассчитываем Uniform baseline…",
    "The Uniform baseline cannot meet this target under the same constraints. No budget difference is available.": "Uniform baseline не достигает цели при тех же ограничениях. Разница бюджетов недоступна.",
    "Calendar KPI": "KPI календарного плана",
    "Uniform KPI": "KPI равномерного плана",
    "Calendar required budget": "Бюджет календарного плана",
    "Uniform required budget": "Бюджет равномерного плана",
    "KPI difference": "Разница KPI",
    "Budget difference": "Разница бюджетов, ₽",
    "Calendar minus Uniform; percentage relative to Uniform.": "Календарный минус равномерный; процент относительно Uniform.",
    "Unit cost is spend per unit of the selected KPI (CPC for clicks, CPA for conversions).": "Цена единицы выбранного KPI: CPC для кликов, CPA для конверсий.",
    "Expected reach": "Ожидаемый охват",
    "Media budget optimization and planning": "Оптимизация медиабюджета и медиапланирование",
    "Allocate media budget across channels and campaign days, accounting for capacity, nonlinear response, and the target KPI.": "Распределение медиабюджета между каналами и днями кампании с учётом ёмкости, нелинейного отклика и целевого KPI.",
    "Type A — Fixed budget": "Тип A — Фиксированный бюджет",
    "Set a budget and horizon to maximize clicks or conversions.": "Задайте бюджет и срок, чтобы получить максимум кликов или конверсий.",
    "Type B — Target KPI": "Тип B — Целевой KPI",
    "Set a target and horizon to find the minimum modeled budget or an explanation of infeasibility.": "Задайте цель и срок, чтобы найти минимальный модельный бюджет или получить объяснение недостижимости.",
    "Core planning flow": "Основной сценарий",
    "Review the catalog (optional)": "Изучить каталог (опционально)",
    "Define a Type A or Type B task": "Задать задачу A или B",
    "Get a media plan or recommendations": "Получить медиаплан или рекомендации",
    "Optional: after campaign launch": "Дополнительно: после запуска кампании",
    "Load or simulate fact, monitor delivery, replan future days, and explore adaptive strategies and experiments.": "Загружайте или моделируйте факт, следите за результатами, перепланируйте будущие дни и исследуйте адаптивные стратегии.",
    "Planner: Overview · Inventory · Media Plan": "Планирование: Обзор · Инвентарь · Медиаплан",
    "Optional: Fact · Monitor · Adaptive Control · Experiments": "Дополнительно: Факт · Мониторинг · Адаптивное управление · Эксперименты",
    "Optional extension": "Дополнительный раздел",
    "Nothing to reset in this session.": "В этой сессии пока нечего сбрасывать.",
    "Confirm planning scenario reset": "Подтверждаю сброс сценария",
    "Public benchmark context + explicit synthetic assumptions → a reproducible planning catalog.": "Открытые ориентиры и явные синтетические допущения → воспроизводимый каталог каналов для планирования.",
    "All catalog parameters": "Все параметры каталога",
    "Selected channel parameters": "Параметры выбранного канала",
    "Planner metrics": "Показатели планирования",
    "Research methods": "Исследовательские методы",
    "Simulator debug / evaluation-only": "Отладка симулятора / только для оценки",
    "Forecast clicks": "Прогноз кликов",
    "Forecast conversions": "Прогноз конверсий",
    "Minimum modeled budget": "Минимальный модельный бюджет",
    "Achieved": "Достигнуто",
    "Detailed channel results": "Подробные результаты по каналам",
    "Whole-plan metrics": "Сводные метрики медиаплана",
    "Detailed allocation explanation": "Подробное объяснение распределения",
    "Uniform baseline comparison": "Сравнение с равномерным baseline",
    "Optional: continue with a campaign": "Дополнительно: продолжить работу с кампанией",
    "The media plan is complete. Optionally approve it as a campaign to load fact, monitor delivery, and replan future days.": "Медиаплан готов. При желании запустите кампанию, чтобы загружать факт, отслеживать результаты и перепланировать будущие дни.",
    "Your plan or infeasibility recommendations are a complete planning result.": "Медиаплан или рекомендации по недостижимой цели — завершённый результат планирования.",
    # Navigation and onboarding
    "Language": "Язык",
    "Media Planning & Campaign Control": "Медиапланирование и управление кампанией",
    "Media Control": "Управление кампанией",
    "Navigate": "Разделы",
    "Overview": "Обзор",
    "Inventory / Generator": "Инвентарь",
    "Media Plan": "Медиаплан",
    "Fact Ingestion": "Загрузка факта",
    "Campaign Monitor": "Мониторинг кампании",
    "Adaptive Control": "Адаптивное управление",
    "Experiments": "Эксперименты",
    "How it works": "Как это работает",
    "No active campaign": "Активной кампании нет",
    "Active: {name}": "Кампания: {name}",
    "Day {day}/{horizon} · {policy}": "День {day}/{horizon} · {policy}",
    "Reset controls": "Сброс и новый сценарий",
    "Confirm campaign reset": "Подтверждаю сброс кампании",
    "Reset campaign": "Сбросить кампанию",
    "Start new planning scenario": "Новый сценарий планирования",
    "Confirm catalog reset": "Подтверждаю сброс каталога",
    "Reset catalog to seed 42": "Вернуть каталог seed 42",
    "Active campaigns retain their own catalog snapshot. Catalog changes affect only new plans.": "Активная кампания хранит свою версию инвентаря. Изменения повлияют только на новые планы.",
    "Start a successful media plan before using campaign controls.": "Сначала рассчитайте медиаплан и запустите кампанию.",
    "Create media plan": "Создать медиаплан",
    "Create new plan": "Создать медиаплан",
    "Inspect inventory": "Посмотреть инвентарь",
    "Run demo": "Запустить демо",
    "Start here": "НАЧАТЬ ЗДЕСЬ",
    "Canonical demo ready: seed 42, ₽1,200,000, 21 days, conversions, all channels. Review inventory, then calculate the plan.": "Демо готово: seed 42, бюджет 1 200 000 ₽, 21 день, конверсии, все каналы. Изучите инвентарь и рассчитайте медиаплан.",
    "Finish or reset the active campaign before starting a new demo.": "Перед новым демо завершите или сбросьте активную кампанию.",
    "Workflow": "Путь пользователя",
    "Current step": "Текущий шаг",
    "Completed": "Готово",
    "Next step": "Следующий шаг",
    "No further step": "Все шаги пройдены",
    "What to do here": "Что делать на этой странице",
    "Metrics and methods": "Показатели и методы",
    "Saturation": "Насыщение",
    "Check inputs: {details}": "Проверьте введённые данные: {details}",
    "Inventory": "Инвентарь",
    "Start Campaign": "Запуск кампании",
    "Fact": "Загрузка факта",
    "Monitoring": "Мониторинг",
    "Replanning": "Перепланирование",
    "Inventory: Inspect the benchmark-derived channels and their response curves.": "Инвентарь: изучите каналы и кривые отклика на основе открытых ориентиров.",
    "Media Plan: Enter a budget or target KPI, campaign horizon, and available channels. Calculate the allocation.": "Медиаплан: задайте бюджет или целевой показатель, срок и доступные каналы. Рассчитайте распределение.",
    "Start Campaign: Start the calculated plan to preserve its original forecast.": "Запуск кампании: запустите рассчитанный план, чтобы сохранить исходный прогноз.",
    "Fact: Simulate delivery for a demo or upload reviewed campaign data in CSV.": "Загрузка факта: сгенерируйте синтетические данные для демо или загрузите проверенный CSV.",
    "Monitoring: Compare the plan in force with fact and inspect the forecast.": "Мониторинг: сравните действующий план с фактом и изучите прогноз.",
    "Replanning: Recalculate only the remaining budget and days.": "Перепланирование: пересчитайте только оставшийся бюджет и будущие дни.",
    "Adaptive Control: Inspect or change the future allocation strategy.": "Адаптивное управление: изучите или смените стратегию распределения на будущие дни.",
    "Experiments: Compare strategies in synthetic scenarios; treat results as illustrative.": "Эксперименты: сравните стратегии в синтетических сценариях. Результаты служат иллюстрацией.",
    "Create a plan, start a campaign, add fact, monitor results, and adjust future spend.": "Рассчитайте план, запустите кампанию, добавьте факт, оцените результат и скорректируйте будущие расходы.",
    "Review channels, then calculate a plan.": "Изучите каналы, затем рассчитайте медиаплан.",
    "After reviewing inventory, calculate the media plan.": "После просмотра инвентаря перейдите к расчёту медиаплана.",
    "After calculation, start the campaign.": "После расчёта запустите кампанию.",
    "Next, add fact by simulation or CSV upload.": "Далее добавьте факт: смоделируйте его или загрузите CSV.",
    "After adding fact, open campaign monitoring.": "После добавления факта откройте мониторинг кампании.",
    "After monitoring, recalculate the remaining plan.": "После мониторинга пересчитайте оставшийся план.",
    "After replanning, inspect adaptive control.": "После перепланирования изучите адаптивное управление.",
    "After adaptive control, inspect experiments.": "После адаптивного управления посмотрите эксперименты.",
    "All steps complete. Review the results or start another scenario.": "Все шаги завершены. Изучите результаты или начните новый сценарий.",
    "Run a comparison or inspect the report.": "Запустите сравнение стратегий или изучите отчёт.",
    "Go to Media Plan →": "Перейти к медиаплану →",
    "Go to fact →": "Добавить факт →",
    "View monitoring →": "Посмотреть мониторинг →",
    "Recalculate remaining plan →": "Пересчитать оставшийся план →",
    "Open experiments →": "Перейти к экспериментам →",
    "Active campaign state": "Состояние кампании",
    "Campaign day": "День кампании",
    "Original budget": "Исходный бюджет",
    "Observed spend": "Фактические расходы",
    "Remaining budget": "Остаток бюджета",
    "Original planned KPI": "Исходный плановый KPI",
    "Observed KPI": "Фактический KPI",
    "Forecast KPI": "Прогноз KPI",
    "Adaptive policy": "Стратегия распределения",
    "Current plan version": "Текущая версия плана",
    "Last replan": "Последний пересчёт",
    "Not yet": "Ещё не было",
    "Projected final KPI minus the original planned KPI.": "Разница между итоговым прогнозом и исходным планом KPI.",
    "Planning assumptions and forecasts are model outputs; observed KPI and spend come only from ingested observed or simulated fact.": "Параметры планирования и прогноз получены из модели. Фактический KPI и расходы поступают только из загруженных или синтетических данных.",
    "Continue campaign": "Продолжить кампанию",
    "Load fact": "Загрузить факт",
    "Open adaptive control": "Открыть адаптивное управление",
    # Inventory
    "BENCHMARK RANGE → GENERATED PLANNING ASSUMPTION. Hidden simulator truth is available only through the explicit debug control below.": "Открытые ориентиры → синтетические параметры планирования. Скрытые параметры симулятора доступны только в отдельном исследовательском режиме.",
    "Catalog seed": "Seed каталога",
    "Regenerate catalog": "Сгенерировать каталог",
    "Generated planning catalog with seed {seed}.": "Каталог с seed {seed} создан.",
    "Generated planning assumptions": "Параметры планирования",
    "Public benchmark context": "Открытые ориентиры",
    "Inspect channel": "Изучить канал",
    "Spend, RUB": "Расходы, ₽",
    "Impressions / contacts": "Показы / контакты",
    "Clicks": "Клики",
    "Conversions": "Конверсии",
    "clicks": "Клики",
    "conversions": "Конверсии",
    "reach": "Охват",
    "Effective CPM, RUB": "Эффективный CPM, ₽",
    "Effective CTR": "Расчётный CTR",
    "Effective CR": "Расчётный CR",
    "KPI response curves": "Отклик KPI на расходы",
    "Nonlinear inventory buyout": "Выкуп инвентаря при росте расходов",
    "Performance degradation under saturation": "Снижение отклика при насыщении",
    "Effective CPM rises because progressively scarcer inventory requires more spend per delivered impression. Effective CTR and CR decline mildly as the available audience is exhausted; together these effects reduce marginal KPI per RUB at higher spend.": "При росте расходов доступного инвентаря остаётся меньше, поэтому цена за тысячу показов растёт. CTR и CR немного снижаются по мере насыщения аудитории, а отдача от дополнительного рубля падает.",
    "Debug: expose simulated hidden truth": "Подробнее: скрытые параметры симулятора",
    "Simulator truth seed": "Seed симулятора",
    "SIMULATED HIDDEN TRUTH — evaluation/debug only; adaptive policies cannot access it.": "СКРЫТЫЕ ПАРАМЕТРЫ СИМУЛЯТОРА — только для оценки; стратегии не получают к ним доступ.",
    # Planning
    "The requested target is infeasible for the selected assumptions.": "Цель недостижима при текущих ограничениях",
    "Requested": "Запрошено",
    "Maximum": "Практический максимум",
    "Gap": "Недостаток",
    "All reach columns mean non-deduplicated expected reach summed across channel-days, not unique campaign reach.": "Охват суммируется по каналам и дням без устранения повторов; это не уникальный охват кампании.",
    "No plan summary returned.": "Сводка медиаплана недоступна.",
    "What can you change?": "Что можно изменить?",
    "Estimate budgets for feasible options": "Оценить бюджет для осуществимых вариантов",
    "Budget estimates may take several seconds; they use the same Type B optimizer and are model estimates, not guarantees.": "Оценка бюджета может занять несколько секунд: используется тот же оптимизатор Type B. Это модельные оценки, а не гарантии.",
    "Estimating counterfactual budgets…": "Расчёт бюджета для альтернатив…",
    "Planned spend": "Плановые расходы",
    "Effective CPM": "Эффективный CPM",
    "Expected unused": "Неиспользованный бюджет",
    "Budget remains unused because every selected channel reached its configured daily spend/inventory limit; forcing more spend would violate the planning assumptions.": "Часть бюджета осталась: выбранные каналы достигли дневных ограничений. Дополнительные расходы нарушили бы параметры планирования.",
    "Budget allocation": "Распределение бюджета",
    "Download channel summary CSV": "Скачать сводку каналов CSV",
    "Daily cumulative prediction": "Прогноз KPI с накоплением по дням",
    "Download daily plan CSV": "Скачать дневной план CSV",
    "Weekly predicted delivery": "Прогноз по неделям",
    "Why this allocation?": "Почему такое распределение?",
    "Explanations use deterministic marginal KPI efficiency, saturation, and spend caps.": "Объяснение основано на отдаче дополнительного рубля, насыщении и ограничениях каналов.",
    "All outputs are the original plan predictions from generated planning assumptions; they are not measured campaign results.": "Все результаты медиаплана — прогноз по синтетическим параметрам, а не измеренные итоги кампании.",
    "Task": "Задача",
    "Planning mode": "Режим планирования",
    "Calendar-aware": "Календарный",
    "Uniform baseline": "Равномерный baseline",
    "Campaign start date": "Дата начала кампании",
    "Calendar-aware mode optimizes each channel-day using deterministic synthetic temporal assumptions, not measured weekday forecasts.": "Календарный режим оптимизирует расходы по каждому каналу и дню с учётом детерминированных синтетических временных предположений. Это не измеренный прогноз по дням недели.",
    "Uniform baseline treats days as exchangeable and splits each channel's optimized spend evenly.": "Равномерный baseline считает все дни одинаковыми и делит оптимизированный бюджет каждого канала поровну.",
    "DETERMINISTIC SYNTHETIC TEMPORAL ASSUMPTIONS — not measured seasonality or platform forecasts.": "Временные коэффициенты синтетические: это не измеренная сезонность и не прогноз платформ.",
    "Daily total spend": "Расходы по дням",
    "Channel × day spend": "Расходы по каналам и дням",
    "Calendar day profiles": "Синтетические дневные профили",
    "Temporal allocation economics": "Экономика распределения по дням",
    "Compare with Uniform baseline": "Сравнить с равномерным baseline",
    "Uniform vs Calendar-aware": "Равномерный vs календарный план",
    "Delta (Calendar − Uniform)": "Разница (календарный − равномерный)",
    "Temporal variation is modeled, not measured; differences are scenario-specific.": "Временная вариация смоделирована, а не измерена; различия зависят от сценария.",
    "A — Maximize KPI": "A — Максимум результата при заданном бюджете",
    "B — Minimum budget for target": "B — Минимальный бюджет для целевого KPI",
    "Horizon, days": "Срок кампании, дней",
    "Objective": "Целевой KPI",
    "Budget, RUB": "Бюджет, ₽",
    "Target metric": "Целевой показатель",
    "Non-deduplicated expected reach": "Ожидаемый охват без устранения повторов",
    "Target value": "Целевое значение",
    "Selected channels": "Доступные каналы",
    "Calculate media plan": "Рассчитать медиаплан",
    "Optimizing…": "Рассчитываем медиаплан…",
    "Campaign name": "Название кампании",
    "Seed 42 conversion campaign": "Демо: конверсии, seed 42",
    "Starting this plan will replace the active session campaign. Export any fact you need before continuing.": "Запуск нового плана заменит текущую кампанию в этой сессии. При необходимости сохраните данные факта.",
    "Confirm active campaign replacement": "Подтверждаю замену кампании",
    "Start campaign": "Запустить кампанию",
    # Fact and monitoring
    "Download campaign observed fact CSV": "Скачать факт кампании CSV",
    "Days simulated": "Смоделировано дней",
    "New fact rows": "Новых строк факта",
    "Simulated spend": "Синтетические расходы",
    "Simulated clicks": "Синтетические клики",
    "Simulated conversions": "Синтетические конверсии",
    "Simulation reached the campaign horizon; no future days remain.": "Симуляция дошла до конца кампании: будущих дней нет.",
    "Simulated Fact": "Синтетический факт",
    "CSV Upload": "Загрузка CSV",
    "SIMULATED OBSERVED FACT — synthetic demo/research data, not real delivery.": "СИНТЕТИЧЕСКИЙ ФАКТ — данные для демо, а не реальные результаты размещения.",
    "Current completed day": "Последний завершённый день",
    "Simulator seed": "Seed симулятора",
    "Hidden parameter deviation": "Отклонение модели от скрытого сценария",
    "Enable weekend and fatigue variation": "Добавить выходные и накопительную усталость аудитории",
    "Simulator settings locked for this campaign: {config}": "Параметры симулятора для кампании зафиксированы: {config}",
    "Adaptive allocation quantum, RUB": "Шаг распределения, ₽",
    "Next day": "Следующий день",
    "Next 3 days": "Следующие 3 дня",
    "Next 7 days": "Следующие 7 дней",
    "Remaining campaign": "До конца кампании",
    "Simulating observed delivery…": "Моделируем результаты…",
    "Simulated fact ingested through day {day} under {policy}.": "Синтетический факт добавлен по день {day} включительно. Стратегия: {policy}.",
    "Current deterministic channel statuses: {statuses}": "Текущие статусы каналов: {statuses}",
    "No channel status is available yet.": "Статусы каналов пока недоступны.",
    "Uploaded rows become OBSERVED FACT after explicit validation and ingestion.": "Загруженные строки станут фактическими данными только после проверки и подтверждения.",
    "Upload complete campaign days. Missing channel/day rows count as zero; the latest uploaded day advances the campaign. Use absolute campaign day for incremental uploads. Reach-target campaigns require reach in every row; simulation does not generate reach.": "Загружайте полные дни кампании. Отсутствующие строки считаются нулевыми; последний загруженный день продвигает кампанию. Для последующих загрузок указывайте абсолютный день. При цели по охвату охват нужен в каждой строке; симулятор его не создаёт.",
    "Download CSV template": "Скачать шаблон CSV",
    "Upload campaign fact CSV": "Загрузить факт кампании CSV",
    "{count} day/channel row(s) already exist in campaign history.": "В истории уже есть {count} строк для этих дней и каналов.",
    "Ingesting this file would exceed the original campaign budget.": "Загрузка превысит исходный бюджет кампании.",
    "Schema, values, and campaign-history validation passed.": "Формат, значения и история кампании проверены.",
    "Validated rows": "Проверено строк",
    "Day range": "Дни",
    "Channels": "Каналы",
    "Uploaded spend": "Загруженные расходы",
    "Ingest observed fact": "Добавить фактические данные",
    "Uploaded fact ingested": "Загруженный факт добавлен",
    "Ingested {count} rows.": "Добавлено строк: {count}.",
    "Forecast final KPI": "Итоговый прогноз KPI",
    "Difference from the original planned final KPI.": "Разница с исходным итоговым планом KPI.",
    "Forecast range (empirical/non-calibrated): {lower}–{upper} from {days} observed day(s).": "Ориентировочный диапазон прогноза: {lower}–{upper} по {days} дням факта. Не калиброванный доверительный интервал.",
    "Forecast range (empirical/non-calibrated): unavailable until at least 3 observed days are present (currently {days}).": "Ориентировочный диапазон появится после 3 дней факта. Сейчас: {days}.",
    "Budget pace": "Темп расходов",
    "KPI pace": "Темп KPI",
    "Observed KPI variance vs plan in force": "Отклонение KPI от действующего плана",
    "Cost per non-deduplicated reach": "Цена неуникального охвата",
    "Reach is non-deduplicated. Status is descriptive, not a significance test; omitted fact rows count as zero through the latest observed day.": "Охват не очищен от повторов. Статус описательный и не означает статистической значимости; пропущенные строки факта считаются нулевыми.",
    "Observed spend is ahead of the active plan. Forecast and static execution proportionally scale remaining allocations to the remaining budget; approved plan history is unchanged. Replan to approve new future allocations.": "Фактические расходы опережают план. Прогноз и исполнение уменьшают будущие расходы пропорционально остатку бюджета. Утверждённая история плана сохраняется; для нового распределения пересчитайте план.",
    "Current {cost}": "Текущий {cost}",
    "Original plan": "Исходный план",
    "Plan in force": "Действующий план",
    "Observed fact": "Фактические данные",
    "Forecast": "Прогноз",
    "Revised plan": "Скорректированный план",
    "Daily spend: original plan / plan in force / observed fact": "Расходы по дням: исходный план, действующий план и факт",
    "Cumulative target KPI: original plan / plan in force / observed fact / forecast": "KPI с накоплением: исходный план, действующий план, факт и прогноз",
    "Download monitoring daily CSV": "Скачать мониторинг по дням CSV",
    "Channel status": "Статус каналов",
    "Status compares observed KPI per RUB with planned KPI per RUB using ±10% thresholds. It is descriptive and does not imply statistical significance.": "Статус сравнивает фактический и плановый KPI на рубль с порогом ±10%. Это описание отклонения, а не проверка статистической значимости.",
    "Download channel monitoring CSV": "Скачать мониторинг каналов CSV",
    "Forecast history": "История прогнозов",
    "Plan versions": "Версии плана",
    "Initial allocation": "Первоначальное распределение",
    "Initial static media plan": "Исходный статический медиаплан",
    "Manual remaining-plan recalculation": "Пересчёт оставшегося плана вручную",
    "Underperforming": "Ниже плана",
    "Outperforming": "Выше плана",
    "On track": "По плану",
    "Spend is capped by the configured daily inventory limit.": "Расходы ограничены дневным лимитом инвентаря.",
    "No additional KPI is expected at this saturation level.": "Дополнительные расходы почти не увеличат KPI при этом насыщении.",
    "Excluded by the request's include/exclude channel selection.": "Канал исключён из выбранного набора.",
    # Adaptive and experiments
    "Static policy": "Без адаптации",
    "Periodic reoptimization": "Периодический пересчёт",
    "Oracle (evaluation-only)": "Oracle (только для оценки)",
    "Adaptive policies act only on the remaining campaign. Observed fact and all earlier plan versions remain immutable.": "Стратегии меняют только будущие расходы. Факт и предыдущие версии плана сохраняются.",
    "Plan version before action": "Версия до действия",
    "Remaining days": "Осталось дней",
    "Forecast before action": "Прогноз до действия",
    "Adaptive policy for remaining campaign": "Стратегия для оставшихся дней",
    "Periodic replan interval, days": "Интервал пересчёта, дней",
    "Switch policy": "Сменить стратегию",
    "Adaptive policy switched to {policy}. Observed fact, spend, and plan history were preserved.": "Стратегия изменена: {policy}. Факт, расходы и история плана сохранены.",
    "Next scheduled replan: before day {day}": "Следующий плановый пересчёт: перед днём {day}",
    "Next scheduled replan: none before completion": "До конца кампании плановый пересчёт не ожидается",
    "Recalculate remaining plan": "Пересчитать оставшийся план",
    "Created revised plan {after} from {before} at day {day}; historical observed fact was unchanged.": "Создан скорректированный план {after} из {before} на дне {day}; история факта сохранена.",
    "Revised plan version": "Версия скорректированного плана",
    "Forecast before replan": "Прогноз до пересчёта",
    "Forecast after replan": "Прогноз после пересчёта",
    "Policy diagnostics": "Диагностика стратегии",
    "Posterior mean and uncertainty summarize observed KPI events per impression; selection also includes current marginal KPI per RUB.": "Среднее постериорной оценки и её неопределённость отражают наблюдаемые события на показ; выбор также учитывает отдачу дополнительного рубля.",
    "LinUCB combines estimated context contribution with an uncertainty bonus. Offline results show material sensitivity to alpha; no absolute superiority is claimed.": "LinUCB сочетает оценку влияния контекста и бонус за неопределённость. Результат чувствителен к alpha и не означает общего превосходства.",
    "Future allocation by plan version": "Будущие расходы по версиям плана",
    "Policy Experiments": "Эксперименты со стратегиями",
    "Compare adaptive policies on paired synthetic scenarios. The Oracle is evaluation-only and is never available to a deployable policy.": "Сравните стратегии в парных синтетических сценариях. Oracle доступен только для оценки и не используется действующими стратегиями.",
    "Paired seeds": "Парных seed",
    "Horizon": "Срок, дней",
    "Quantum, RUB": "Шаг распределения, ₽",
    "LinUCB alpha": "Параметр LinUCB alpha",
    "Run comparison": "Сравнить стратегии",
    "Running paired policy simulations…": "Сравниваем стратегии…",
    "Configure a bounded comparison and run it explicitly.": "Укажите параметры и нажмите «Сравнить стратегии».",
    "Download experiment summary CSV": "Скачать сводку эксперимента CSV",
    "Download experiment runs CSV": "Скачать запуски эксперимента CSV",
    "Download experiment config JSON": "Скачать параметры эксперимента JSON",
    "KPI distribution": "Распределение KPI",
    "Adaptive policy": "Стратегия распределения",
    "Mean regret to Oracle (evaluation-only)": "Среднее отставание от Oracle (только для оценки)",
    "Mean cumulative KPI": "Средний KPI с накоплением",
    "Paired KPI difference versus static": "Разница KPI со стратегией без адаптации",
    "Oracle (evaluation-only) uses simulator truth and is not a deployable policy. Simulator truth is synthetic. Results depend on planner misspecification, horizon, noise, context strength, and hyperparameters. Periodic replanning can be competitive; exploration can hurt with accurate priors or short campaigns; LinUCB alpha sensitivity is nontrivial.": "Oracle использует скрытые синтетические параметры только для оценки. Результаты зависят от ошибок модели, срока, шума, контекста и настроек. Периодический пересчёт может быть конкурентоспособен; исследование каналов иногда снижает результат, особенно при короткой кампании. LinUCB чувствителен к alpha.",
    # Short help text
    "help.CPM": "Цена за тысячу показов. В модели растёт при выкупе более редкого инвентаря.",
    "help.CPC": "Расходы на один клик: расходы / клики.",
    "help.CPA": "Расходы на одну конверсию: расходы / конверсии.",
    "help.CTR": "Доля показов, которые привели к клику: клики / показы.",
    "help.CR": "Доля кликов, которые привели к конверсии: конверсии / клики.",
    "help.VTR": "Доля показов видео, которые привели к просмотру.",
    "help.saturation": "Степень выкупа доступного инвентаря. При насыщении отдача от дополнительных расходов снижается.",
    "help.oracle": "Oracle знает скрытые параметры симулятора и нужен только для сравнения. Это не стратегия для реальной кампании.",
    "help.thompson": "Thompson Sampling случайно выбирает оценки эффективности с учётом накопленных наблюдений и неопределённости.",
    "help.linucb": "LinUCB учитывает контекст и добавляет бонус каналам с высокой неопределённостью.",
}

EN_HELP = {
    "help.CPM": "Cost per thousand impressions. The model's effective CPM rises as scarce inventory is bought.",
    "help.CPC": "Cost per click: spend divided by clicks.",
    "help.CPA": "Cost per conversion: spend divided by conversions.",
    "help.CTR": "Click-through rate: clicks divided by impressions.",
    "help.CR": "Post-click conversion rate: conversions divided by clicks.",
    "help.VTR": "Video view-through rate: views divided by eligible video impressions.",
    "help.saturation": "Share of available inventory bought. Incremental returns decline as saturation rises.",
    "help.oracle": "Oracle knows hidden simulator parameters for evaluation only. It cannot control a real campaign.",
    "help.thompson": "Thompson Sampling samples channel performance from uncertainty-aware estimates learned from observations.",
    "help.linucb": "LinUCB uses observable context and an uncertainty bonus when comparing channels.",
}

TABLE_RU = {
    "parameter": "Параметр", "generated_value": "Значение", "unit": "Единица",
    "meaning": "Что означает", "benchmark-derived/model range": "Ориентир / диапазон",
    "source category": "Источник", "category": "Категория", "metric": "Показатель",
    "published_range": "Публичный ориентир", "model_range": "Диапазон модели",
    "source_name": "Название источника", "channel": "Канал", "day": "День", "week": "Неделя",
    "start_day": "Первый день", "end_day": "Последний день", "spend": "Расходы",
    "weekday": "День недели", "kpi": "KPI", "daily_cap": "Дневной лимит расходов",
    "supply_factor": "Коэффициент доступности", "cpm_factor": "Коэффициент CPM",
    "ctr_factor": "Коэффициент CTR", "cr_factor": "Коэффициент CR",
    "max_spend_factor": "Коэффициент лимита расходов",
    "mode": "Режим", "unit_cost": "Стоимость единицы KPI",
    "daily_spend_sd": "Отклонение дневных расходов",
    "planned_spend": "Плановые расходы", "actual_spend": "Фактические расходы",
    "spend_delta": "Разница расходов", "spend_pace": "Темп расходов",
    "planned_kpi": "Плановый KPI", "actual_kpi": "Фактический KPI",
    "kpi_delta": "Разница KPI", "status": "Статус",
    "impressions": "Показы", "reach": "Ожидаемый охват (сумма без устранения повторов)",
    "non_deduplicated_reach": "Охват без устранения повторов",
    "clicks": "Клики", "conversions": "Конверсии", "video_views": "Просмотры видео",
    "ctr": "CTR", "cr": "CR", "vtr": "VTR", "cpm": "CPM", "cpc": "CPC", "cpa": "CPA",
    "date": "Дата", "source": "Тип факта", "policy": "Код стратегии", "policy_label": "Стратегия",
    "plan_version": "Версия плана", "plan_policy": "Стратегия плана",
    "version": "Версия", "version_type": "Тип плана", "created_at_day": "Создан на дне",
    "reason": "Причина", "remaining_budget": "Остаток бюджета",
    "remaining_horizon": "Осталось дней", "forecast_at_creation": "Прогноз при создании",
    "main_allocation_delta": "Главное изменение",
    "inventory_saturation": "Насыщение инвентаря",
    "marginal_kpi_per_rub": "Предельная отдача, KPI/₽",
    "relative_marginal_efficiency": "Относительная отдача",
    "efficiency_rank": "Место по отдаче", "explanation": "Причина распределения",
    "horizon_spend_cap": "Лимит расходов", "eligible": "Доступен",
    "cumulative_target_kpi": "Целевой KPI с накоплением", "cumulative_kpi": "KPI с накоплением",
    "original_planned_spend": "Исходные плановые расходы",
    "original_planned_kpi": "Исходный плановый KPI",
    "planned_cumulative_spend": "Плановые расходы с накоплением",
    "actual_cumulative_spend": "Фактические расходы с накоплением",
    "planned_cumulative_kpi": "Плановый KPI с накоплением",
    "actual_cumulative_kpi": "Фактический KPI с накоплением",
    "planned_ctr": "Плановый CTR", "actual_ctr": "Фактический CTR",
    "planned_cr": "Плановый CR", "actual_cr": "Фактический CR",
    "planned_unit_cost": "Плановая цена KPI", "actual_unit_cost": "Фактическая цена KPI",
    "unit_cost_delta": "Разница цены KPI", "planned_cpa": "Плановый CPA",
    "actual_cpa": "Фактический CPA", "cpa_delta": "Разница CPA",
    "ctr_delta": "Разница CTR", "cr_delta": "Разница CR",
    "updated_ctr": "Обновлённый CTR", "updated_cr": "Обновлённый CR",
    "original_ctr": "Исходный CTR", "original_cr": "Исходный CR",
    "posterior_mean": "Средняя оценка", "uncertainty": "Неопределённость",
    "context_contribution": "Вклад контекста", "exploration_bonus": "Бонус исследования",
    "score": "Оценка", "rank": "Место", "recommended": "Рекомендуется",
    "future_spend": "Будущие расходы", "before_replan_spend": "До пересчёта, ₽",
    "after_replan_spend": "После пересчёта, ₽", "allocation_shift": "Изменение, ₽",
    "runs": "Запусков", "seed": "Seed", "mean_kpi": "Средний KPI",
    "std_kpi": "Стандартное отклонение", "median_kpi": "Медиана KPI",
    "q25_kpi": "25-й процентиль", "q75_kpi": "75-й процентиль",
    "mean_cpc": "Средний CPC", "mean_cpa": "Средний CPA",
    "mean_regret_to_oracle": "Среднее отставание от Oracle",
    "mean_delta_vs_static": "Средняя разница с базовой стратегией",
    "wins_vs_static": "Выигрышей у базовой стратегии",
    "requested_target": "Запрошенная цель", "total_kpi": "Итоговый KPI", "regret_to_oracle": "Отставание от Oracle",
    "kpi_delta_vs_static": "Разница с базовой стратегией",
}

STATUS_RU = {
    "Underperforming": RU["Underperforming"],
    "Outperforming": RU["Outperforming"],
    "On track": RU["On track"],
    "upload": "Загружен",
    "simulator": "Синтетический",
    "Original plan": RU["Original plan"],
    "Revised plan": RU["Revised plan"],
}

PARAMETER_RU = {
    "contacts/day": "контактов/день", "expected people/day": "ожидаемых людей/день",
    "RUB / 1,000": "₽ / 1 000", "RUB/day": "₽/день",
    "impressions/person": "показов/человек", "fraction": "доля", "RUB": "₽",
    "Daily buyable supply": "Доступный объём закупки за день",
    "Potential daily audience pool; achieved reach also depends on impressions and frequency": "Потенциальная дневная аудитория; достижимый охват также зависит от показов и частоты",
    "Low-spend marginal inventory price": "Предельная цена инвентаря при малых расходах",
    "Configured daily spend cap": "Установленный дневной лимит расходов",
    "Expected frequency used in reach response": "Ожидаемая частота показов в модели охвата",
    "Unsaturated click-through assumption": "Базовое предположение о CTR без насыщения",
    "Unsaturated post-click conversion assumption": "Базовое предположение о CR после клика без насыщения",
    "Video view-through assumption, when applicable": "Предположение о доле видеопросмотров, если применимо",
    "CTR loss at full inventory saturation": "Снижение CTR при полном насыщении инвентаря",
    "CR loss at full inventory saturation": "Снижение CR при полном насыщении инвентаря",
    "Derived buyout scale retained for compatibility": "Производный масштаб выкупа инвентаря",
    "derived from capacity and base CPM": "выведено из ёмкости и базового CPM",
    "not applicable": "не применимо",
    "social and programmatic/native media": "социальная и программатик/нативная реклама",
    "marketplace sponsored advertising": "реклама в маркетплейсах",
    "SMS": "SMS",
    "synthetic model assumption / derived value": "синтетическое предположение модели / производное значение",
}

WORKFLOW_STEPS = (
    "Inventory", "Media Plan", "Start Campaign", "Fact", "Monitoring",
    "Replanning", "Adaptive Control", "Experiments",
)


@dataclass(frozen=True)
class WorkflowProgress:
    completed: tuple[bool, ...]
    current_index: int | None

    @property
    def next_step(self) -> str | None:
        return None if self.current_index is None else WORKFLOW_STEPS[self.current_index]


def workflow_progress(
    *, inventory_reviewed: bool, plan_ready: bool, campaign_started: bool,
    fact_loaded: bool, monitor_reviewed: bool, replanned: bool,
    adaptive_opened: bool, experimented: bool,
) -> WorkflowProgress:
    """Presentation progress from recorded actions, without changing business state."""
    completed = (
        inventory_reviewed or plan_ready or campaign_started,
        plan_ready or campaign_started,
        campaign_started,
        fact_loaded,
        monitor_reviewed and fact_loaded,
        replanned,
        adaptive_opened and replanned,
        experimented,
    )
    current = next((index for index, done in enumerate(completed) if not done), None)
    return WorkflowProgress(completed, current)


def translate(key: str, language: str = "ru", **values: object) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported UI language: {language}")
    if language == "ru":
        template = RU[key]
    else:
        template = EN_HELP[key] if key.startswith("help.") else key
    return template.format(**values)


def table_columns(language: str) -> dict[str, str]:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported UI language: {language}")
    return TABLE_RU if language == "ru" else {}


def localized_columns(columns, language: str) -> list[str]:
    """Validate display headers before Streamlit/PyArrow sees the dataframe."""
    labels = table_columns(language)
    source_columns = list(columns)
    rendered = [labels.get(column, column) for column in source_columns]
    duplicates = {
        label: [source for source, translated in zip(source_columns, rendered) if translated == label]
        for label in rendered if rendered.count(label) > 1
    }
    if duplicates:
        raise ValueError(f"Duplicate localized dataframe columns ({language}): {duplicates}")
    return rendered


def localize_frame(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    """Translate a display copy; analytical columns and exported values stay intact."""
    columns = localized_columns(frame.columns, language)
    displayed = frame.copy()
    for names, label in (
        (("weekday",), weekday_label),
        (("meaning", "unit", "source category", "benchmark-derived/model range"), parameter_label),
        (("status", "source", "version_type"), status_label),
        (("explanation",), allocation_explanation),
        (("reason", "main_allocation_delta"), reason_label),
    ):
        for name in names:
            if name in displayed:
                displayed[name] = displayed[name].map(
                    lambda value: label(value if name == "weekday" else str(value), language)
                    if pd.notna(value) else value
                )
    displayed.columns = columns
    return displayed


def status_label(value: str, language: str) -> str:
    return STATUS_RU.get(value, value) if language == "ru" else value


def parameter_label(value: str, language: str) -> str:
    return PARAMETER_RU.get(value, value) if language == "ru" else value


def weekday_label(value: int, language: str) -> str:
    days = (
        ("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье")
        if language == "ru" else
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    )
    return days[int(value)]


def feasibility_text(
    feasibility, metric: str, language: str, horizon_days: int | None = None
) -> tuple[str, list[str]]:
    """Render a structured feasibility result without parsing solver prose."""
    if language == "en":
        return feasibility.explanation, list(feasibility.recommendations)
    metric_ru = {"clicks": "кликов", "conversions": "конверсий", "reach": "контактов охвата"}.get(metric, metric)
    explanation = (
        f"Запрошено {feasibility.requested_value:,.2f} {metric_ru}, но практический максимум "
        f"для выбранных каналов и срока — {feasibility.maximum_achievable:,.2f}."
    )
    recommendations: list[str] = []
    if (feasibility.additional_capacity_with_all_channels is not None
            and feasibility.maximum_with_all_channels is not None
            and feasibility.maximum_with_all_channels >= feasibility.requested_value):
        recommendations.append(
            f"Подключите все каналы: модельная ёмкость вырастет с "
            f"{feasibility.maximum_achievable:,.2f} до "
            f"{feasibility.maximum_with_all_channels:,.2f} {metric_ru}; "
            "этого достаточно для исходной цели."
        )
    if feasibility.minimum_feasible_horizon is not None:
        start = f"с {horizon_days} " if horizon_days is not None else ""
        option = (
            f"Увеличьте срок {start}до {feasibility.minimum_feasible_horizon} дней "
            f"(+{feasibility.additional_days_needed}) при выбранных каналах."
        )
        if feasibility.estimated_budget_at_minimum_horizon is not None:
            option += f" Оценка минимального бюджета: {feasibility.estimated_budget_at_minimum_horizon:,.2f} ₽."
        recommendations.append(option)
    if feasibility.recommended_target is not None:
        at_horizon = f"при сроке {horizon_days} дней " if horizon_days is not None else "при текущем сроке "
        option = (
            f"Снизьте цель {at_horizon}до примерно "
            f"{feasibility.recommended_target:,.2f} {metric_ru}."
        )
        if feasibility.estimated_budget_for_recommended_target is not None:
            option += f" Оценка минимального бюджета: {feasibility.estimated_budget_for_recommended_target:,.2f} ₽."
        recommendations.append(option)
    return explanation, recommendations


def reason_label(value: str, language: str) -> str:
    if language == "en":
        return value
    if value in RU:
        return RU[value]
    if value.startswith("Observed day "):
        return "Факт за день " + value.removeprefix("Observed day ")
    if value.startswith("Scheduled replan before day "):
        return "Плановый пересчёт перед днём " + value.removeprefix("Scheduled replan before day ")
    return value


def allocation_explanation(value: str, language: str) -> str:
    """Translate solver-derived reason text for display, keeping its numbers."""
    if language == "en":
        return value
    if value in RU:
        return RU[value]
    prefix = "Not funded: initial marginal efficiency is "
    suffix = " of the strongest current alternative."
    if value.startswith(prefix) and value.endswith(suffix):
        fraction = value[len(prefix):-len(suffix)]
        return f"Канал не получил бюджет: начальная отдача составляет {fraction} от лучшего доступного канала."
    prefix = "Funded while marginal KPI per RUB remains competitive (current efficiency rank "
    if value.startswith(prefix) and value.endswith(")."):
        rank = value[len(prefix):-2]
        return f"Канал финансируется благодаря конкурентной отдаче дополнительного рубля (место {rank})."
    raise KeyError(f"Untranslated allocation explanation: {value}")

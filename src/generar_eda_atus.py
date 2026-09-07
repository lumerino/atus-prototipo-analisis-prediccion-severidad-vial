"""Copia vendorizada de `actividad2/generar_eda_atus.py` para el prototipo del
Entregable 4: `generar_modelado_atus.py` (también vendorizado en esta carpeta)
depende de `clean`, `norm_code`, `severity_from_clasacc`, `to_int`, `write_csv`
y las listas `HEAVY_VEHICLE_COLUMNS`/`VEHICLE_COLUMNS`/`VICTIM_COLUMNS` definidas
aquí. El resto del archivo (streaming de los 28 CSV crudos, generación de
figuras EDA) no se ejecuta desde el prototipo; sus constantes de ruta
(`BASE_DIR`, `DATA_ROOT`, `DATA_DIR`, `CATALOG_DIR`, `OUT_DIR`) no son válidas
en esta ubicación ni se usan.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "actividad1" / "atus_zip"
DATA_DIR = DATA_ROOT / "conjunto_de_datos"
CATALOG_DIR = DATA_ROOT / "catalogos"
OUT_DIR = BASE_DIR / "resultados_eda"

NUMERIC_COLUMNS = [
    "ID_HORA",
    "ID_MINUTO",
    "ID_DIA",
    "ID_EDAD",
    "AUTOMOVIL",
    "CAMPASAJ",
    "MICROBUS",
    "PASCAMION",
    "OMNIBUS",
    "TRANVIA",
    "CAMIONETA",
    "CAMION",
    "TRACTOR",
    "FERROCARRI",
    "MOTOCICLET",
    "BICICLETA",
    "OTROVEHIC",
    "CONDMUERTO",
    "CONDHERIDO",
    "PASAMUERTO",
    "PASAHERIDO",
    "PEATMUERTO",
    "PEATHERIDO",
    "CICLMUERTO",
    "CICLHERIDO",
    "OTROMUERTO",
    "OTROHERIDO",
    "NEMUERTO",
    "NEHERIDO",
]

VEHICLE_COLUMNS = [
    "AUTOMOVIL",
    "CAMPASAJ",
    "MICROBUS",
    "PASCAMION",
    "OMNIBUS",
    "TRANVIA",
    "CAMIONETA",
    "CAMION",
    "TRACTOR",
    "FERROCARRI",
    "MOTOCICLET",
    "BICICLETA",
    "OTROVEHIC",
]

HEAVY_VEHICLE_COLUMNS = ["MICROBUS", "PASCAMION", "OMNIBUS", "CAMION", "TRACTOR", "FERROCARRI"]

VICTIM_COLUMNS = [
    "CONDMUERTO",
    "CONDHERIDO",
    "PASAMUERTO",
    "PASAHERIDO",
    "PEATMUERTO",
    "PEATHERIDO",
    "CICLMUERTO",
    "CICLHERIDO",
    "OTROMUERTO",
    "OTROHERIDO",
    "NEMUERTO",
    "NEHERIDO",
]

SPECIAL_CODES = {
    "ID_HORA": {"99": "Hora no especificada"},
    "ID_MINUTO": {"99": "Minuto no especificado"},
    "ID_DIA": {"32": "Dia no especificado"},
    "ID_EDAD": {"0": "Conductor se fugo", "99": "Edad ignorada"},
    "DIASEMANA": {"No especificado": "Dia no especificado", "Certificado cero": "Certificado cero"},
    "CLASACC": {"Certificado cero": "Excluir de modelado"},
}


def clean(value: object) -> str:
    return str(value or "").strip()


def norm_code(value: object, width: int) -> str:
    text = clean(value)
    return text.zfill(width) if text.isdigit() else text


def to_int(value: object) -> int | None:
    text = clean(value)
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def strip_accents_for_class(value: str) -> str:
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def severity_from_clasacc(clasacc: str) -> int | None:
    text = strip_accents_for_class(clean(clasacc)).lower()
    if text in {"fatal", "no fatal"}:
        return 1
    if text in {"solo daños", "solo danos"}:
        return 0
    return None


def percentile_from_counts(counter: Counter, total: int, p: float) -> float | None:
    if total == 0:
        return None
    target = max(1, int(total * p + 0.999999))
    cumulative = 0
    for value in sorted(counter):
        cumulative += counter[value]
        if cumulative >= target:
            return float(value)
    return float(max(counter))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_entity_catalog() -> dict[str, str]:
    catalog_path = CATALOG_DIR / "tc_entidad.csv"
    entities: dict[str, str] = {}
    with catalog_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entities[norm_code(row["ID_ENTIDAD"], 2)] = clean(row["NOM_ENTIDAD"])
    return entities


def read_municipio_catalog() -> dict[tuple[str, str], str]:
    catalog_path = CATALOG_DIR / "tc_municipio.csv"
    municipios: dict[tuple[str, str], str] = {}
    with catalog_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (norm_code(row["ID_ENTIDAD"], 2), norm_code(row["ID_MUNICIPIO"], 3))
            municipios[key] = clean(row["NOM_MUNICIPIO"])
    return municipios


def zona_from_urbana_suburbana(urbana: str, suburbana: str) -> str:
    urbana = clean(urbana)
    suburbana = clean(suburbana)
    if urbana and urbana != "Sin accidente en esta zona":
        return f"Urbana: {urbana}"
    if suburbana and suburbana != "Sin accidente en esta zona":
        return f"Suburbana: {suburbana}"
    return "No especificado"


def make_summary_markdown(
    total_records: int,
    records_by_year: Counter,
    class_counts: Counter,
    top_entities: list[dict[str, object]],
    top_entity_severity: list[dict[str, object]],
) -> None:
    total_valid = sum(class_counts[k] for k in ("Fatal", "No fatal", "Sólo daños") if k in class_counts)
    victims = class_counts.get("Fatal", 0) + class_counts.get("No fatal", 0)
    only_damage = class_counts.get("Sólo daños", 0)
    severity_rate = victims / total_valid if total_valid else 0

    lines = [
        "# Resumen EDA ATUS",
        "",
        "Este archivo fue generado automáticamente por `generar_eda_atus.py` para apoyar el borrador del Entregable 2.",
        "",
        "## Cifras principales",
        "",
        f"- Registros procesados: {total_records:,}",
        f"- Años cubiertos: {min(records_by_year)}-{max(records_by_year)}",
        f"- Accidentes válidos para severidad binaria: {total_valid:,}",
        f"- Accidentes con víctimas: {victims:,}",
        f"- Accidentes de sólo daños: {only_damage:,}",
        f"- Proporción de accidentes con víctimas: {severity_rate:.2%}",
        "",
        "## Entidades con más registros",
        "",
        "| Entidad | Accidentes totales |",
        "|---|---:|",
    ]
    for row in top_entities[:10]:
        lines.append(f"| {row['entidad']} | {int(row['accidentes_totales']):,} |")

    lines.extend([
        "",
        "## Entidades con mayor proporción de accidentes con víctimas",
        "",
        "| Entidad | Accidentes válidos | Accidentes con víctimas | Proporción con víctimas |",
        "|---|---:|---:|---:|",
    ])
    for row in top_entity_severity[:10]:
        lines.append(
            f"| {row['entidad']} | {int(row['accidentes_validos']):,} | "
            f"{int(row['accidentes_con_victimas']):,} | {float(row['proporcion_con_victimas']):.2%} |"
        )

    (OUT_DIR / "resumen_eda_atus.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entities = read_entity_catalog()
    municipios = read_municipio_catalog()
    files = sorted(DATA_DIR.glob("atus_anual_*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {DATA_DIR}")

    records_by_year: Counter = Counter()
    valid_by_year: Counter = Counter()
    victims_by_year: Counter = Counter()
    class_counts: Counter = Counter()
    month_counts: Counter = Counter()
    weekday_counts: Counter = Counter()
    hour_counts: Counter = Counter()
    weekday_hour_counts: Counter = Counter()
    time_band_counts: Counter = Counter()
    entity_counts: Counter = Counter()
    entity_valid_counts: Counter = Counter()
    entity_victim_counts: Counter = Counter()
    municipio_counts: Counter = Counter()
    municipio_valid_counts: Counter = Counter()
    municipio_victim_counts: Counter = Counter()
    caparod_counts: Counter = Counter()
    caparod_valid_counts: Counter = Counter()
    caparod_victim_counts: Counter = Counter()
    zona_counts: Counter = Counter()
    zona_valid_counts: Counter = Counter()
    zona_victim_counts: Counter = Counter()
    conductor_counts: Counter = Counter()
    conductor_valid_counts: Counter = Counter()
    conductor_victim_counts: Counter = Counter()
    tipaccid_counts: Counter = Counter()
    tipaccid_valid_counts: Counter = Counter()
    tipaccid_victim_counts: Counter = Counter()
    cause_counts: Counter = Counter()
    cause_valid_counts: Counter = Counter()
    cause_victim_counts: Counter = Counter()
    special_counts: Counter = Counter()
    numeric_counts: dict[str, Counter] = {col: Counter() for col in NUMERIC_COLUMNS}
    numeric_sums: defaultdict[str, int] = defaultdict(int)
    numeric_square_sums: defaultdict[str, int] = defaultdict(int)
    clean_age_counts: Counter = Counter()
    clean_age_sum = 0
    clean_age_square_sum = 0
    total_vehicle_counts: Counter = Counter()
    vehicle_indicator_counts: Counter = Counter()
    negative_value_counts: Counter = Counter()
    total_records = 0

    for file_path in files:
        file_records = 0
        with file_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_records += 1
                file_records += 1
                year = clean(row.get("ANIO")) or file_path.stem[-4:]
                records_by_year[year] += 1

                clasacc = clean(row.get("CLASACC"))
                class_counts[clasacc] += 1
                severity = severity_from_clasacc(clasacc)

                entity_id = norm_code(row.get("ID_ENTIDAD"), 2)
                entity_counts[entity_id] += 1
                municipio_id = (entity_id, norm_code(row.get("ID_MUNICIPIO"), 3))
                municipio_counts[municipio_id] += 1
                if severity is not None:
                    valid_by_year[year] += 1
                    if severity == 1:
                        victims_by_year[year] += 1
                    entity_valid_counts[entity_id] += 1
                    if severity == 1:
                        entity_victim_counts[entity_id] += 1
                    municipio_valid_counts[municipio_id] += 1
                    if severity == 1:
                        municipio_victim_counts[municipio_id] += 1

                caparod = clean(row.get("CAPAROD"))
                caparod_counts[caparod] += 1
                if severity is not None:
                    caparod_valid_counts[caparod] += 1
                    if severity == 1:
                        caparod_victim_counts[caparod] += 1

                zona = zona_from_urbana_suburbana(row.get("URBANA"), row.get("SUBURBANA"))
                zona_counts[zona] += 1
                if severity is not None:
                    zona_valid_counts[zona] += 1
                    if severity == 1:
                        zona_victim_counts[zona] += 1

                for driver_col in ("ALIENTO", "CINTURON", "SEXO"):
                    driver_key = (driver_col, clean(row.get(driver_col)))
                    conductor_counts[driver_key] += 1
                    if severity is not None:
                        conductor_valid_counts[driver_key] += 1
                        if severity == 1:
                            conductor_victim_counts[driver_key] += 1

                month_counts[norm_code(row.get("MES"), 2)] += 1
                weekday_counts[clean(row.get("DIASEMANA"))] += 1
                hour = to_int(row.get("ID_HORA"))
                hour_key = "99" if hour == 99 else f"{hour:02d}" if hour is not None else ""
                hour_counts[hour_key] += 1
                weekday_hour_counts[(clean(row.get("DIASEMANA")), hour_key)] += 1
                if hour is None:
                    band = "Sin dato"
                elif hour == 99:
                    band = "No especificada"
                elif 0 <= hour <= 5:
                    band = "Madrugada"
                elif 6 <= hour <= 11:
                    band = "Mañana"
                elif 12 <= hour <= 17:
                    band = "Tarde"
                elif 18 <= hour <= 23:
                    band = "Noche"
                else:
                    band = "Fuera de rango"
                time_band_counts[band] += 1

                tipaccid = clean(row.get("TIPACCID"))
                tipaccid_counts[tipaccid] += 1
                if severity is not None:
                    tipaccid_valid_counts[tipaccid] += 1
                    if severity == 1:
                        tipaccid_victim_counts[tipaccid] += 1

                cause = clean(row.get("CAUSAACCI"))
                cause_counts[cause] += 1
                if severity is not None:
                    cause_valid_counts[cause] += 1
                    if severity == 1:
                        cause_victim_counts[cause] += 1

                for col, codes in SPECIAL_CODES.items():
                    value = clean(row.get(col))
                    if value in codes:
                        special_counts[(col, value, codes[value])] += 1

                for col in NUMERIC_COLUMNS:
                    value = to_int(row.get(col))
                    if value is not None:
                        numeric_counts[col][value] += 1
                        numeric_sums[col] += value
                        numeric_square_sums[col] += value * value
                        if value < 0:
                            negative_value_counts[col] += 1

                vehicle_values = {col: to_int(row.get(col)) or 0 for col in VEHICLE_COLUMNS}
                total_vehicles = sum(vehicle_values.values())
                total_vehicle_counts[total_vehicles] += 1
                if vehicle_values.get("MOTOCICLET", 0) > 0:
                    vehicle_indicator_counts["involucra_motocicleta"] += 1
                if vehicle_values.get("BICICLETA", 0) > 0:
                    vehicle_indicator_counts["involucra_bicicleta"] += 1
                if any(vehicle_values.get(col, 0) > 0 for col in HEAVY_VEHICLE_COLUMNS):
                    vehicle_indicator_counts["involucra_pesado"] += 1

                age = to_int(row.get("ID_EDAD"))
                if age is not None and 1 <= age <= 98:
                    clean_age_counts[age] += 1
                    clean_age_sum += age
                    clean_age_square_sum += age * age
        print(f"Procesado {file_path.name}: {file_records:,} registros", flush=True)

    records_rows = []
    for year in sorted(records_by_year):
        count = records_by_year[year]
        records_rows.append(
            {
                "anio": year,
                "registros": count,
                "proporcion_total": round(count / total_records, 6),
                "porcentaje_total": f"{count / total_records * 100:.2f}%",
            }
        )
    write_csv(OUT_DIR / "tabla_02_registros_por_anio.csv", records_rows, ["anio", "registros", "proporcion_total", "porcentaje_total"])

    yearly_severity_rows = []
    for year in sorted(records_by_year):
        valid_count = valid_by_year[year]
        victims = victims_by_year[year]
        yearly_severity_rows.append(
            {
                "anio": year,
                "accidentes_totales": records_by_year[year],
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    write_csv(OUT_DIR / "serie_severidad_por_anio.csv", yearly_severity_rows, ["anio", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"])

    class_rows = []
    for key, count in class_counts.most_common():
        class_rows.append(
            {
                "clase_accidente": key,
                "registros": count,
                "porcentaje_total": round(count / total_records * 100, 4),
                "uso_modelado": "Excluir" if key == "Certificado cero" else "Clase positiva" if severity_from_clasacc(key) == 1 else "Clase negativa" if severity_from_clasacc(key) == 0 else "Revisar",
            }
        )
    write_csv(OUT_DIR / "tabla_06_distribucion_clasacc.csv", class_rows, ["clase_accidente", "registros", "porcentaje_total", "uso_modelado"])

    special_rows = [
        {
            "variable": col,
            "codigo_valor": code,
            "interpretacion": label,
            "frecuencia": count,
            "porcentaje_total": round(count / total_records * 100, 4),
        }
        for (col, code, label), count in sorted(special_counts.items())
    ]
    write_csv(OUT_DIR / "tabla_04_codigos_especiales.csv", special_rows, ["variable", "codigo_valor", "interpretacion", "frecuencia", "porcentaje_total"])

    numeric_rows = []
    for col, counts in numeric_counts.items():
        total = sum(counts.values())
        if not counts:
            continue
        numeric_rows.append(
            {
                "variable": col,
                "conteo": total,
                "media": round(numeric_sums[col] / total, 4),
                "desv_estandar": round(math.sqrt((numeric_square_sums[col] / total) - (numeric_sums[col] / total) ** 2), 4),
                "mediana": round(percentile_from_counts(counts, total, 0.50), 4),
                "min": min(counts),
                "p25": round(percentile_from_counts(counts, total, 0.25), 4),
                "p75": round(percentile_from_counts(counts, total, 0.75), 4),
                "p90": round(percentile_from_counts(counts, total, 0.90), 4),
                "p95": round(percentile_from_counts(counts, total, 0.95), 4),
                "p99": round(percentile_from_counts(counts, total, 0.99), 4),
                "max": max(counts),
            }
        )
    write_csv(OUT_DIR / "tabla_05_resumen_estadistico.csv", numeric_rows, ["variable", "conteo", "media", "desv_estandar", "mediana", "min", "p25", "p75", "p90", "p95", "p99", "max"])

    total_vehicle_total = sum(total_vehicle_counts.values())
    total_vehicle_sum = sum(value * count for value, count in total_vehicle_counts.items())
    total_vehicle_square_sum = sum(value * value * count for value, count in total_vehicle_counts.items())
    derived_rows = [
        {
            "variable": "total_vehiculos",
            "conteo": total_vehicle_total,
            "media": round(total_vehicle_sum / total_vehicle_total, 4),
            "desv_estandar": round(math.sqrt((total_vehicle_square_sum / total_vehicle_total) - (total_vehicle_sum / total_vehicle_total) ** 2), 4),
            "mediana": round(percentile_from_counts(total_vehicle_counts, total_vehicle_total, 0.50), 4),
            "min": min(total_vehicle_counts),
            "p25": round(percentile_from_counts(total_vehicle_counts, total_vehicle_total, 0.25), 4),
            "p75": round(percentile_from_counts(total_vehicle_counts, total_vehicle_total, 0.75), 4),
            "p90": round(percentile_from_counts(total_vehicle_counts, total_vehicle_total, 0.90), 4),
            "p95": round(percentile_from_counts(total_vehicle_counts, total_vehicle_total, 0.95), 4),
            "p99": round(percentile_from_counts(total_vehicle_counts, total_vehicle_total, 0.99), 4),
            "max": max(total_vehicle_counts),
        }
    ]
    clean_age_total = sum(clean_age_counts.values())
    if clean_age_total:
        derived_rows.append(
            {
                "variable": "edad_limpia_1_98",
                "conteo": clean_age_total,
                "media": round(clean_age_sum / clean_age_total, 4),
                "desv_estandar": round(math.sqrt((clean_age_square_sum / clean_age_total) - (clean_age_sum / clean_age_total) ** 2), 4),
                "mediana": round(percentile_from_counts(clean_age_counts, clean_age_total, 0.50), 4),
                "min": min(clean_age_counts),
                "p25": round(percentile_from_counts(clean_age_counts, clean_age_total, 0.25), 4),
                "p75": round(percentile_from_counts(clean_age_counts, clean_age_total, 0.75), 4),
                "p90": round(percentile_from_counts(clean_age_counts, clean_age_total, 0.90), 4),
                "p95": round(percentile_from_counts(clean_age_counts, clean_age_total, 0.95), 4),
                "p99": round(percentile_from_counts(clean_age_counts, clean_age_total, 0.99), 4),
                "max": max(clean_age_counts),
            }
        )
    write_csv(OUT_DIR / "tabla_resumen_derivadas.csv", derived_rows, ["variable", "conteo", "media", "desv_estandar", "mediana", "min", "p25", "p75", "p90", "p95", "p99", "max"])

    vehicle_indicator_rows = []
    for indicator in ["involucra_motocicleta", "involucra_bicicleta", "involucra_pesado"]:
        count = vehicle_indicator_counts[indicator]
        vehicle_indicator_rows.append(
            {
                "indicador": indicator,
                "registros": count,
                "porcentaje_total": round(count / total_records * 100, 4),
            }
        )
    write_csv(OUT_DIR / "tabla_indicadores_vehiculares.csv", vehicle_indicator_rows, ["indicador", "registros", "porcentaje_total"])

    total_vehicle_rows = [
        {
            "total_vehiculos": value,
            "registros": count,
            "porcentaje_total": round(count / total_records * 100, 4),
        }
        for value, count in sorted(total_vehicle_counts.items())
    ]
    write_csv(OUT_DIR / "distribucion_total_vehiculos.csv", total_vehicle_rows, ["total_vehiculos", "registros", "porcentaje_total"])

    negative_rows = [
        {
            "variable": col,
            "registros_negativos": count,
            "porcentaje_total": round(count / total_records * 100, 6),
        }
        for col, count in sorted(negative_value_counts.items())
    ]
    write_csv(OUT_DIR / "tabla_valores_negativos.csv", negative_rows, ["variable", "registros_negativos", "porcentaje_total"])

    entity_rows = []
    for entity_id, count in entity_counts.most_common():
        entity_rows.append(
            {
                "id_entidad": entity_id,
                "entidad": entities.get(entity_id, entity_id),
                "accidentes_totales": count,
                "porcentaje_total": round(count / total_records * 100, 4),
            }
        )
    write_csv(OUT_DIR / "tabla_07_top_entidades_total.csv", entity_rows, ["id_entidad", "entidad", "accidentes_totales", "porcentaje_total"])

    min_entity_records = 10000
    entity_severity_rows = []
    for entity_id, valid_count in entity_valid_counts.items():
        if valid_count < min_entity_records:
            continue
        victims = entity_victim_counts[entity_id]
        entity_severity_rows.append(
            {
                "id_entidad": entity_id,
                "entidad": entities.get(entity_id, entity_id),
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6),
            }
        )
    entity_severity_rows.sort(key=lambda row: row["proporcion_con_victimas"], reverse=True)
    write_csv(OUT_DIR / "tabla_08_top_entidades_severidad.csv", entity_severity_rows, ["id_entidad", "entidad", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"])

    # Códigos de municipio no catalogados (no existen en tc_municipio.csv): se excluyen de los
    # rankings de municipio (tablas 09 y 10) para no distorsionar los hallazgos territoriales,
    # pero permanecen en los conteos generales (entity_counts, totales nacionales, etc.).
    # Ver tabla_municipios_no_catalogados.csv para el detalle de lo excluido.
    uncatalogued_rows = []
    for (entity_id, municipio_id), count in municipio_counts.items():
        if (entity_id, municipio_id) in municipios:
            continue
        valid_count = municipio_valid_counts.get((entity_id, municipio_id), 0)
        victims = municipio_victim_counts.get((entity_id, municipio_id), 0)
        uncatalogued_rows.append(
            {
                "id_entidad": entity_id,
                "id_municipio": municipio_id,
                "entidad": entities.get(entity_id, entity_id),
                "accidentes_totales": count,
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
            }
        )
    uncatalogued_rows.sort(key=lambda row: row["accidentes_totales"], reverse=True)
    write_csv(
        OUT_DIR / "tabla_municipios_no_catalogados.csv",
        uncatalogued_rows,
        ["id_entidad", "id_municipio", "entidad", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas"],
    )
    excluded_total = sum(row["accidentes_totales"] for row in uncatalogued_rows)
    excluded_valid = sum(row["accidentes_validos"] for row in uncatalogued_rows)
    print(
        f"Municipios no catalogados excluidos de tablas 09/10: {len(uncatalogued_rows):,} pares, "
        f"{excluded_total:,} registros totales, {excluded_valid:,} registros válidos."
    )

    municipio_rows = []
    for (entity_id, municipio_id), count in municipio_counts.items():
        if (entity_id, municipio_id) not in municipios:
            continue
        municipio_rows.append(
            {
                "id_entidad": entity_id,
                "id_municipio": municipio_id,
                "entidad": entities.get(entity_id, entity_id),
                "municipio": municipios.get((entity_id, municipio_id), municipio_id),
                "accidentes_totales": count,
                "porcentaje_total": round(count / total_records * 100, 4),
            }
        )
    municipio_rows.sort(key=lambda row: row["accidentes_totales"], reverse=True)
    write_csv(
        OUT_DIR / "tabla_09_top_municipios_total.csv",
        municipio_rows[:15],
        ["id_entidad", "id_municipio", "entidad", "municipio", "accidentes_totales", "porcentaje_total"],
    )

    # Umbral más bajo que el usado para entidades (10,000) porque los municipios son unidades más pequeñas.
    min_municipio_records = 3000
    municipio_severity_rows = []
    for (entity_id, municipio_id), valid_count in municipio_valid_counts.items():
        if valid_count < min_municipio_records:
            continue
        if (entity_id, municipio_id) not in municipios:
            continue
        victims = municipio_victim_counts[(entity_id, municipio_id)]
        municipio_severity_rows.append(
            {
                "id_entidad": entity_id,
                "id_municipio": municipio_id,
                "entidad": entities.get(entity_id, entity_id),
                "municipio": municipios.get((entity_id, municipio_id), municipio_id),
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    municipio_severity_rows.sort(key=lambda row: row["proporcion_con_victimas"], reverse=True)
    write_csv(
        OUT_DIR / "tabla_10_top_municipios_severidad.csv",
        municipio_severity_rows[:15],
        ["id_entidad", "id_municipio", "entidad", "municipio", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"],
    )

    caparod_rows = []
    for key, count in caparod_counts.most_common():
        valid_count = caparod_valid_counts[key]
        victims = caparod_victim_counts[key]
        caparod_rows.append(
            {
                "caparod": key,
                "accidentes_totales": count,
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    write_csv(OUT_DIR / "tabla_caparod_severidad.csv", caparod_rows, ["caparod", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"])

    zona_rows = []
    for key, count in zona_counts.most_common():
        valid_count = zona_valid_counts[key]
        victims = zona_victim_counts[key]
        zona_rows.append(
            {
                "zona": key,
                "accidentes_totales": count,
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    write_csv(OUT_DIR / "tabla_zona_severidad.csv", zona_rows, ["zona", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"])

    conductor_variable_order = {"ALIENTO": 0, "CINTURON": 1, "SEXO": 2}
    conductor_rows = []
    for (variable, categoria), count in conductor_counts.items():
        valid_count = conductor_valid_counts[(variable, categoria)]
        victims = conductor_victim_counts[(variable, categoria)]
        conductor_rows.append(
            {
                "variable": variable,
                "categoria": categoria,
                "accidentes_totales": count,
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    conductor_rows.sort(key=lambda row: (conductor_variable_order.get(row["variable"], 99), -row["accidentes_totales"]))
    write_csv(
        OUT_DIR / "tabla_conductor_severidad.csv",
        conductor_rows,
        ["variable", "categoria", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"],
    )

    def count_rows(counter: Counter, name: str) -> list[dict[str, object]]:
        return [
            {name: key, "registros": count, "porcentaje_total": round(count / total_records * 100, 4)}
            for key, count in counter.most_common()
        ]

    write_csv(OUT_DIR / "serie_accidentes_por_mes.csv", count_rows(month_counts, "mes"), ["mes", "registros", "porcentaje_total"])
    write_csv(OUT_DIR / "serie_accidentes_por_dia_semana.csv", count_rows(weekday_counts, "dia_semana"), ["dia_semana", "registros", "porcentaje_total"])
    write_csv(OUT_DIR / "serie_accidentes_por_hora.csv", count_rows(hour_counts, "hora"), ["hora", "registros", "porcentaje_total"])
    write_csv(OUT_DIR / "serie_accidentes_por_franja_horaria.csv", count_rows(time_band_counts, "franja_horaria"), ["franja_horaria", "registros", "porcentaje_total"])
    weekday_hour_rows = [
        {
            "dia_semana": weekday,
            "hora": hour_key,
            "registros": count,
            "porcentaje_total": round(count / total_records * 100, 4),
        }
        for (weekday, hour_key), count in sorted(weekday_hour_counts.items())
    ]
    write_csv(OUT_DIR / "matriz_dia_semana_hora.csv", weekday_hour_rows, ["dia_semana", "hora", "registros", "porcentaje_total"])

    tipaccid_rows = []
    for key, count in tipaccid_counts.most_common():
        valid_count = tipaccid_valid_counts[key]
        victims = tipaccid_victim_counts[key]
        tipaccid_rows.append(
            {
                "tipo_accidente": key,
                "accidentes_totales": count,
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    write_csv(OUT_DIR / "tabla_tipo_accidente_severidad.csv", tipaccid_rows, ["tipo_accidente", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"])

    cause_rows = []
    for key, count in cause_counts.most_common():
        valid_count = cause_valid_counts[key]
        victims = cause_victim_counts[key]
        cause_rows.append(
            {
                "causa_presunta": key,
                "accidentes_totales": count,
                "accidentes_validos": valid_count,
                "accidentes_con_victimas": victims,
                "proporcion_con_victimas": round(victims / valid_count, 6) if valid_count else "",
            }
        )
    write_csv(OUT_DIR / "tabla_causa_presunta_severidad.csv", cause_rows, ["causa_presunta", "accidentes_totales", "accidentes_validos", "accidentes_con_victimas", "proporcion_con_victimas"])

    make_summary_markdown(total_records, records_by_year, class_counts, entity_rows, entity_severity_rows)
    print(f"Procesados {total_records:,} registros de {len(files)} archivos.")
    print(f"Resultados escritos en: {OUT_DIR}")


if __name__ == "__main__":
    main()

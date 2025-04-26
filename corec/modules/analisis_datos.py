import logging
import time
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any
from corec.core import ComponenteBase

class ModuloAnalisisDatos(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAnalisisDatos")
        self.nucleus = None
        self.config = None

    async def inicializar(self, nucleus, config: Dict[str, Any] = None):
        """Inicializa el módulo de análisis de datos."""
        self.nucleus = nucleus
        self.config = config or {}
        self.logger.info("[AnálisisDatos] Módulo inicializado")

    async def analizar(self, df: pd.DataFrame, nombre_dataset: str) -> Dict[str, Any]:
        """Analiza un dataset y calcula estadísticas, correlaciones y anomalías."""
        try:
            t0 = time.monotonic()
            result = {"dataset": nombre_dataset, "estadisticas": {}, "correlaciones": {}, "anomalias": {}}

            # Asegurar que df tenga al menos una columna numérica
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns
            if len(num_cols) == 0:
                self.logger.warning(f"[AnálisisDatos] No hay columnas numéricas en {nombre_dataset}")
                return result

            # Estadísticas descriptivas
            stats = df[num_cols].describe().to_dict()
            result["estadisticas"] = stats

            # Correlaciones (si hay más de una columna numérica)
            if len(num_cols) > 1:
                corr = df[num_cols].corr().to_dict()
                threshold = self.config.get("correlation_threshold", 0.8)
                result["correlaciones"] = {
                    k: {k2: v2 for k2, v2 in v.items() if abs(v2) > threshold and k2 != k}
                    for k, v in corr.items()
                }

            # Detección de anomalías
            iso = IsolationForest(
                n_estimators=self.config.get("n_estimators", 100),
                max_samples=self.config.get("max_samples", 1000),
                contamination=0.1,
                random_state=42
            )
            mask_inliers = iso.fit_predict(df[num_cols]) == 1
            result["anomalias"] = {
                "num_anomalias": len(df) - sum(mask_inliers),
                "indices_anomalias": df.index[~mask_inliers].tolist()
            }

            t1 = time.monotonic()
            await self.nucleus.publicar_alerta({
                "tipo": "analisis_datos",
                "dataset": nombre_dataset,
                "num_filas": len(df),
                "num_anomalias": result["anomalias"]["num_anomalias"],
                "latencia_ms": (t1 - t0) * 1000,
                "timestamp": time.time()
            })

            self.logger.info(f"[AnálisisDatos] Dataset {nombre_dataset} analizado: "
                             f"{len(df)} filas, {result['anomalias']['num_anomalias']} anomalías")
            return result

        except Exception as e:
            self.logger.error(f"[AnálisisDatos] Error analizando {nombre_dataset}: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_analisis_datos",
                "dataset": nombre_dataset,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            return {"dataset": nombre_dataset, "error": str(e)}

    async def detener(self):
        """Detiene el módulo de análisis de datos."""
        self.logger.info("[AnálisisDatos] Módulo detenido")

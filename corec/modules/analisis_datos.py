# corec/modules/analisis_datos.py

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from corec.core import ComponenteBase

class ModuloAnalisisDatos(ComponenteBase):
    """
    Módulo para hacer estadística descriptiva, detección de anomalías
    y extracción de correlaciones fuertes sobre DataFrames.
    """

    def __init__(self):
        self.logger = logging.getLogger("ModuloAnalisisDatos")
        self.nucleus = None
        self.config = {}

    async def inicializar(self, nucleus, config=None):
        """
        Inicializa el módulo.
        config puede contener:
          - iforest_params: dict de parámetros para IsolationForest
          - corr_threshold: float umbral de correlación (p.ej. 0.8)
        """
        self.nucleus = nucleus
        self.config = config or {}
        self.logger.info(f"[AnalisisDatos] Inicializado con config: {self.config}")

    async def analizar(self, df: pd.DataFrame, nombre: str):
        """
        Ejecuta:
          1) Estadísticas descriptivas
          2) Detección de outliers con IsolationForest
          3) Correlaciones absolutas >= corr_threshold
        Publica alertas con cada resultado y devuelve un resumen.
        """
        # 1) Estadísticas
        stats = df.describe().to_dict()
        await self.nucleus.publicar_alerta({
            "tipo": "estadisticas_generadas",
            "nombre": nombre,
            "stats": stats
        })

        # 2) Outliers
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            iso = IsolationForest(**self.config.get("iforest_params", {}))
            mask_inliers = iso.fit_predict(df[num_cols]) == 1
            n_outliers = int((~mask_inliers).sum())
        else:
            n_outliers = 0

        await self.nucleus.publicar_alerta({
            "tipo": "anomalías_detectadas",
            "nombre": nombre,
            "num_anomalías": n_outliers
        })

        # 3) Correlaciones
        corr = df[num_cols].corr().abs() if num_cols else pd.DataFrame()
        umbral = self.config.get("corr_threshold", 0.8)
        pares = [
            (i, j, float(corr.loc[i, j]))
            for i in corr.columns for j in corr.columns
            if i < j and corr.loc[i, j] >= umbral
        ]
        await self.nucleus.publicar_alerta({
            "tipo": "correlaciones_altas",
            "nombre": nombre,
            "pares": pares
        })

        self.logger.info(f"[AnalisisDatos] Análisis completo para {nombre}")
        return {
            "estadisticas": stats,
            "num_anomalías": n_outliers,
            "correlaciones": pares
        }

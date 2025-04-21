async def reparar(self):
    """Repara el bloque simbiótico reactivando entidades inactivas."""
    error_msg = None
    for entidad in self.entidades:
        if getattr(entidad, "estado", None) == "inactiva":  # Verificamos si el atributo existe
            try:
                entidad.estado = "activa"
                self.logger.info(f"[Bloque {self.id}] Entidad {entidad.id} reactivada")
                self.fallos = 0
                await self.nucleus.publicar_alerta({
                    "tipo": "bloque_reparado",
                    "bloque_id": self.id,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"[Bloque {self.id}] Error al reactivar entidad {entidad.id}: {str(e)}")
                error_msg = str(e)
                if self.nucleus:
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_reparacion",
                        "bloque_id": self.id,
                        "mensaje": str(e),
                        "timestamp": time.time()
                    })
                raise  # Relanzamos la excepción para que el test la capture

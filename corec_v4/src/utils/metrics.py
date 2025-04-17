from prometheus_client import Gauge

corec_celu_count = Gauge('corec_celu_count', 'Number of CeluEntidades', ['instance_id'])
corec_micro_count = Gauge('corec_micro_count', 'Number of MicroCeluEntidades', ['instance_id'])
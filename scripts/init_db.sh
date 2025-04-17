#!/bin/bash
docker cp schema.sql corec_v4-postgres-1:/schema.sql
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -f /schema.sql
echo "Database initialized"
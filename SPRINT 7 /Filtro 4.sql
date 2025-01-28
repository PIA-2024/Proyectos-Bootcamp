
# Recupera los identificadores de los barrios de O'Hare y Loop de la tabla neighborhoods.

SELECT 
    neighborhood_id, 
    name 
FROM 
    neighborhoods 
WHERE 
     name LIKE '%Hare' OR name LIKE 'Loop'

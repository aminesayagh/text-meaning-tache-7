# Documentation Pipeline de Traitement de Texte

## Architecture Générale

Notre pipeline de traitement de texte est composée de 5 couches principales :

1. **Ingestion des Données (Kafka)**
2. **Traitement (Spark)**
3. **Stockage (AWS S3)**
4. **Service de Modèles (TensorFlow Serving)**
5. **Monitoring (Prometheus/Grafana)**

## Flux de Données

1. **Ingestion des Données**
   - Les textes entrent dans le système via Kafka
   - 3 topics principaux : `raw-text-fr`, `raw-text-en`, `raw-text-ar`
   - Format des messages :

```json
{
    "text": "contenu du texte",
    "language": "fr/en/ar",
    "metadata": {},
    "timestamp": "2025-01-18T20:00:00Z"
}
```

1. **Traitement avec Spark**
   - Lecture en streaming depuis Kafka
   - Traitement par lots (batch) toutes les minutes
   - Sauvegarde dans S3 avec partitionnement par langue

2. **Stockage dans S3**
   Structure des buckets :

```
s3://raw-bucket/
    ├── raw/
    │   ├── fr/
    │   ├── en/
    │   └── ar/
s3://processed-bucket/
    ├── processed/
    │   ├── language=fr/
    │   ├── language=en/
    │   └── language=ar/
```

3. **Service des Modèles**
   - API REST : `http://localhost:8501`
   - Endpoints principaux :
     - Prédiction : `/v1/models/text_classifier/predict`
     - Status : `/v1/models/text_classifier`
     - Métadonnées : `/v1/models/text_classifier/metadata`

## Guide de Démarrage

1. **Lancement des Services**

   ```bash
   docker-compose up -d
   ```

2. **Vérification des Services**
   - Kafka UI : http://localhost:8080
   - Spark UI : http://localhost:8081
   - TensorFlow Serving : http://localhost:8501
   - Grafana : http://localhost:3000

## Points d'Accès des Services

1. **Kafka**
   - Bootstrap Server : `kafka:29092`
   - Topics :
     - `raw-text-fr` - textes français
     - `raw-text-en` - textes anglais
     - `raw-text-ar` - textes arabes
     - `processed-text` - textes traités

2. **Spark**
   - Master : `spark://spark-master:7077`
   - Web UI : `http://localhost:8081`
   - 2 workers configurés

3. **TensorFlow Serving**
   - REST API : `http://localhost:8501`
   - gRPC : `localhost:8500`
   - Proxy NGINX : `http://localhost:8502`

4. **Monitoring**
   - Prometheus : `http://localhost:9090`
   - Grafana : `http://localhost:3000`

## Surveillance et Maintenance

1. **Logs des Conteneurs**
   ```bash
   # Voir les logs d'un service
   docker-compose logs -f [service]
   ```

2. **Métriques Importantes**
   - Latence de traitement Kafka
   - Utilisation mémoire Spark
   - Temps de réponse des modèles
   - Taux de succès des prédictions

3. **Points de Contrôle**
   - Status des topics Kafka
   - État des workers Spark
   - Disponibilité des modèles
   - Espace de stockage S3

## Procédures de Reprise

1. **En Cas de Panne Kafka**
```bash
   docker-compose restart kafka
   # Vérifier les topics
   docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

2. **En Cas de Problème Spark**

```bash
   docker-compose restart spark-master spark-worker-1 spark-worker-2
   # Relancer les jobs
   docker exec spark-master /opt/spark-apps/submit-job.sh
```

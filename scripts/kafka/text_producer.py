
from kafka import KafkaProducer
import json
from typing import Dict, Any
import logging

class TextProducer:
    def __init__(self, bootstrap_servers: str = 'kafka:29092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topics = {
            'fr': 'raw-text-fr',
            'en': 'raw-text-en',
            'ar': 'raw-text-ar'
        }
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def send_text(self, text: str, language: str, metadata: Dict[Any, Any] = None) -> None:
        """
        Send text data to appropriate Kafka topic based on language
        """
        if language not in self.topics:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {list(self.topics.keys())}")

        topic = self.topics[language]
        
        # Prepare the message
        message = {
            'text': text,
            'language': language,
            'metadata': metadata or {},
            'timestamp': str(datetime.datetime.now())
        }

        try:
            # Send the message
            future = self.producer.send(topic, value=message, key=str(uuid.uuid4()))
            # Wait for the message to be delivered
            record_metadata = future.get(timeout=10)
            
            self.logger.info(
                f"Successfully sent message to topic {topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
        except Exception as e:
            self.logger.error(f"Error sending message to Kafka: {str(e)}")
            raise

    def close(self):
        """
        Close the producer connection
        """
        self.producer.close()

if __name__ == "__main__":
    # Example usage
    producer = TextProducer()
    try:
        # Example: Send a test message for each language
        test_messages = {
            'fr': 'Exemple de texte en français',
            'en': 'Example text in English',
            'ar': 'مثال نص باللغة العربية'
        }
        
        for lang, text in test_messages.items():
            producer.send_text(
                text=text,
                language=lang,
                metadata={'source': 'test', 'category': 'example'}
            )
    finally:
        producer.close()
import asyncio
import json
from aiokafka import AIOKafkaProducer

async def main():
    producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
    await producer.start()
    
    try:
        # Sunucuya gönderilecek START komutu
        command = {
            "client_id": "test_client_01",
            "action": "START",
            "config": {
                "tick_rate": 1.5,      # 1.5 saniyede bir veri üretecek
                "machine_count": 2,    # 2 adet makine için veri üretecek
                
                # Daha hızlı anomali (hata) oluşması için fiziksel parametreleri eziyoruz:
                "stress_multiplier": 50.0,            # Normalden 200 kat daha hızlı eskir (Varsayılan: 0.25)
                "failure_threshold": 10.0,            # Hasar toleransı
                "imbalance_probability": 0.5,         # %50 ihtimalle dengesizlik arızası
                "bearing_fault_probability": 0.5,     # %50 ihtimalle rulman arızası
                "fault_activation_health_threshold": 0.99  # Sağlık %99'un altına düştüğü an arıza başlar
            }
        }
        
        print("Sunucuya START komutu gönderiliyor...")
        print(json.dumps(command, indent=2))
        
        await producer.send_and_wait("simulation-commands", json.dumps(command).encode('utf-8'))
        
        print("\n✅ Komut başarıyla Kafka'ya iletildi!")
        print("Şimdi simulation_server.py'nin çalıştığı terminale bakarak logları görebilirsiniz.")
        print("Veri akışını canlı izlemek için tarayıcıdan http://localhost:8081 adresindeki Kafka UI'a girebilirsiniz.\n")
        print("Durdurmak (STOP) istediğinizde, aynı client_id ile 'action': 'STOP' komutu göndermeniz yeterlidir.")
        
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(main())

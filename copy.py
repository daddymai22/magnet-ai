#!/usr/bin/env python
"""
Magnet AI - Instance Cloner & Distributed Intelligence System

Clones the current AI instance and synchronizes shared learning memory
across all instances, enabling distributed training and knowledge sharing.
"""

import json
import shutil
import subprocess
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import requests
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AIInstance:
    """Represents an AI instance"""
    instance_id: str
    instance_name: str
    version: str
    created_at: str
    location: str  # "local" or "remote"
    model_path: str
    status: str  # "active", "training", "offline"
    last_sync: Optional[str] = None
    memory_hash: Optional[str] = None

class AIInstanceCloner:
    """Clones AI instances and manages distributed copies"""
    
    def __init__(self, base_dir: str = "./"):
        self.base_dir = Path(base_dir)
        self.instances_dir = self.base_dir / "ai_instances"
        self.instances_dir.mkdir(exist_ok=True)
        self.instances_registry = self._load_registry()
        logger.info(f"🤖 AI Instance Cloner initialized")
    
    def _load_registry(self) -> Dict:
        """Load instances registry"""
        registry_file = self.instances_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        return {
            "instances": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_registry(self) -> None:
        """Save instances registry"""
        registry_file = self.instances_dir / "registry.json"
        self.instances_registry["last_updated"] = datetime.now().isoformat()
        with open(registry_file, 'w') as f:
            json.dump(self.instances_registry, f, indent=2)
    
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID"""
        import uuid
        return f"instance_{uuid.uuid4().hex[:8]}"
    
    def _hash_memory(self, memory_data: Dict) -> str:
        """Create hash of memory state for change detection"""
        memory_str = json.dumps(memory_data, sort_keys=True)
        return hashlib.sha256(memory_str.encode()).hexdigest()[:16]
    
    def clone_instance(self, source_name: str = "primary") -> AIInstance:
        """Clone current AI instance locally"""
        logger.info(f"\n🔄 Cloning AI instance '{source_name}'...")
        
        instance_id = self._generate_instance_id()
        instance_dir = self.instances_dir / instance_id
        instance_dir.mkdir(exist_ok=True)
        
        # Clone source files
        files_to_clone = [
            "training.py",
            "environment.py",
            "chat_interface.py",
            "feedback_system.py",
            "training_data_manager.py",
            "config.yaml",
            "requirements.txt"
        ]
        
        for file in files_to_clone:
            src = self.base_dir / file
            if src.exists():
                shutil.copy2(src, instance_dir / file)
        
        # Clone trained models
        logs_dir = self.base_dir / "logs"
        if logs_dir.exists():
            models_dir = instance_dir / "models"
            models_dir.mkdir(exist_ok=True)
            for model_file in logs_dir.glob("*.zip"):
                shutil.copy2(model_file, models_dir / model_file.name)
        
        # Clone training data
        training_data_src = self.base_dir / "training_data.json"
        if training_data_src.exists():
            shutil.copy2(training_data_src, instance_dir / "training_data.json")
        
        # Create instance metadata
        instance = AIInstance(
            instance_id=instance_id,
            instance_name=f"{source_name}_clone_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            version="1.0",
            created_at=datetime.now().isoformat(),
            location="local",
            model_path=str(instance_dir / "models"),
            status="active"
        )
        
        # Save instance metadata
        metadata_file = instance_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(instance), f, indent=2)
        
        # Register instance
        self.instances_registry["instances"].append(asdict(instance))
        self._save_registry()
        
        logger.info(f"✅ Instance cloned: {instance.instance_name}")
        logger.info(f"   Location: {instance_dir}")
        logger.info(f"   Instance ID: {instance_id}")
        
        return instance
    
    def list_instances(self) -> List[AIInstance]:
        """List all registered instances"""
        instances = []
        for inst_data in self.instances_registry["instances"]:
            instances.append(AIInstance(**inst_data))
        return instances
    
    def get_instance(self, instance_id: str) -> Optional[AIInstance]:
        """Get specific instance"""
        for inst_data in self.instances_registry["instances"]:
            if inst_data["instance_id"] == instance_id:
                return AIInstance(**inst_data)
        return None

class DistributedMemorySync:
    """Synchronizes memory across AI instances"""
    
    def __init__(self):
        self.cloner = AIInstanceCloner()
        self.sync_interval = 300  # 5 minutes
        self.sync_thread = None
        logger.info("📡 Distributed Memory Sync initialized")
    
    def _load_training_data(self, instance_dir: Path) -> Dict:
        """Load training data from instance"""
        data_file = instance_dir / "training_data.json"
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_training_data(self, instance_dir: Path, data: Dict) -> None:
        """Save training data to instance"""
        data_file = instance_dir / "training_data.json"
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def merge_shared_memory(self, instance1_data: Dict, instance2_data: Dict) -> Dict:
        """Merge shared memory from two instances"""
        logger.info("🔀 Merging shared memory...")
        
        merged = instance1_data.copy()
        
        # Merge global insights
        insights1 = set(json.dumps(i, sort_keys=True) for i in merged.get("shared_memory", {}).get("global_insights", []))
        insights2 = set(json.dumps(i, sort_keys=True) for i in instance2_data.get("shared_memory", {}).get("global_insights", []))
        all_insights = insights1 | insights2
        
        merged["shared_memory"]["global_insights"] = [json.loads(i) for i in all_insights]
        
        # Merge best techniques
        for tech_key in ["best_hider_techniques", "best_seeker_techniques"]:
            tech1 = set(json.dumps(t, sort_keys=True) for t in merged.get("shared_memory", {}).get(tech_key, []))
            tech2 = set(json.dumps(t, sort_keys=True) for t in instance2_data.get("shared_memory", {}).get(tech_key, []))
            all_techs = tech1 | tech2
            merged["shared_memory"][tech_key] = [json.loads(t) for t in all_techs]
        
        # Merge anomaly patterns
        patterns1 = set(json.dumps(p, sort_keys=True) for p in merged.get("shared_memory", {}).get("anomaly_patterns", []))
        patterns2 = set(json.dumps(p, sort_keys=True) for p in instance2_data.get("shared_memory", {}).get("anomaly_patterns", []))
        all_patterns = patterns1 | patterns2
        merged["shared_memory"]["anomaly_patterns"] = [json.loads(p) for p in all_patterns]
        
        logger.info(f"✅ Merged data:")
        logger.info(f"   - Global insights: {len(merged['shared_memory']['global_insights'])}")
        logger.info(f"   - Best techniques: {len(merged['shared_memory']['best_hider_techniques']) + len(merged['shared_memory']['best_seeker_techniques'])}")
        logger.info(f"   - Anomaly patterns: {len(merged['shared_memory']['anomaly_patterns'])}")
        
        return merged
    
    def sync_instance_pair(self, instance1_id: str, instance2_id: str) -> bool:
        """Sync memory between two instances"""
        logger.info(f"\n🔄 Syncing instances: {instance1_id} <-> {instance2_id}")
        
        try:
            inst1_dir = self.cloner.instances_dir / instance1_id
            inst2_dir = self.cloner.instances_dir / instance2_id
            
            # Load both instances' data
            data1 = self._load_training_data(inst1_dir)
            data2 = self._load_training_data(inst2_dir)
            
            # Merge
            merged = self.merge_shared_memory(data1, data2)
            
            # Save merged data back to both
            self._save_training_data(inst1_dir, merged)
            self._save_training_data(inst2_dir, merged)
            
            logger.info(f"✅ Sync completed between {instance1_id} and {instance2_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
            return False
    
    def sync_all_instances(self) -> None:
        """Synchronize all connected instances"""
        instances = self.cloner.list_instances()
        logger.info(f"\n🌐 Syncing all {len(instances)} instances...")
        
        if len(instances) < 2:
            logger.warning("Need at least 2 instances to sync")
            return
        
        # Sync in pairs
        for i in range(0, len(instances) - 1, 2):
            self.sync_instance_pair(instances[i].instance_id, instances[i+1].instance_id)
    
    def start_continuous_sync(self, sync_interval: int = 300) -> None:
        """Start continuous synchronization in background"""
        logger.info(f"📡 Starting continuous sync (interval: {sync_interval}s)")
        
        def sync_loop():
            while True:
                try:
                    self.sync_all_instances()
                    time.sleep(sync_interval)
                except Exception as e:
                    logger.error(f"Sync loop error: {e}")
                    time.sleep(10)  # Wait before retry
        
        self.sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("✅ Sync thread started")
    
    def stop_continuous_sync(self) -> None:
        """Stop continuous synchronization"""
        if self.sync_thread:
            logger.info("🛑 Stopping sync thread...")
            # Thread is daemon, will exit when main thread exits

class RemoteInstanceManager:
    """Manages remote AI instances"""
    
    def __init__(self, remote_servers: List[str] = None):
        self.remote_servers = remote_servers or []
        logger.info(f"🌐 Remote Instance Manager initialized with {len(self.remote_servers)} servers")
    
    def push_instance_to_remote(self, instance_id: str, server_url: str) -> bool:
        """Push instance to remote server"""
        logger.info(f"📤 Pushing instance {instance_id} to {server_url}...")
        
        try:
            cloner = AIInstanceCloner()
            instance_dir = cloner.instances_dir / instance_id
            
            # Create tarball
            import tarfile
            tar_path = Path(f"/tmp/{instance_id}.tar.gz")
            
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(instance_dir, arcname=instance_id)
            
            # Upload via POST request
            with open(tar_path, 'rb') as f:
                files = {'instance': f}
                response = requests.post(
                    f"{server_url}/api/instances/upload",
                    files=files,
                    timeout=30
                )
            
            tar_path.unlink()  # Clean up
            
            if response.status_code == 200:
                logger.info(f"✅ Instance pushed successfully")
                return True
            else:
                logger.error(f"❌ Upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Push failed: {e}")
            return False
    
    def pull_instance_from_remote(self, instance_id: str, server_url: str) -> bool:
        """Pull instance from remote server"""
        logger.info(f"📥 Pulling instance {instance_id} from {server_url}...")
        
        try:
            response = requests.get(
                f"{server_url}/api/instances/{instance_id}/download",
                timeout=30
            )
            
            if response.status_code == 200:
                import tarfile
                tar_path = Path(f"/tmp/{instance_id}.tar.gz")
                
                with open(tar_path, 'wb') as f:
                    f.write(response.content)
                
                cloner = AIInstanceCloner()
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(cloner.instances_dir)
                
                tar_path.unlink()  # Clean up
                logger.info(f"✅ Instance pulled successfully")
                return True
            else:
                logger.error(f"❌ Download failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pull failed: {e}")
            return False

def main():
    """Demo: Clone instance and setup distributed sync"""
    print("\n" + "="*60)
    print("🤖 MAGNET AI - DISTRIBUTED INTELLIGENCE SYSTEM")
    print("="*60 + "\n")
    
    # Initialize systems
    cloner = AIInstanceCloner()
    sync = DistributedMemorySync()
    
    # Clone the primary instance
    logger.info("\n📋 STEP 1: Cloning Primary Instance")
    instance1 = cloner.clone_instance("primary")
    
    # Create a second clone
    logger.info("\n📋 STEP 2: Creating Second Instance")
    instance2 = cloner.clone_instance("primary")
    
    # List all instances
    logger.info("\n📋 STEP 3: Registered Instances")
    instances = cloner.list_instances()
    for i, inst in enumerate(instances, 1):
        print(f"{i}. {inst.instance_name}")
        print(f"   ID: {inst.instance_id}")
        print(f"   Location: {inst.location}")
        print(f"   Models: {inst.model_path}\n")
    
    # Sync instances
    logger.info("\n📋 STEP 4: Syncing Instances")
    if len(instances) >= 2:
        sync.sync_instance_pair(instances[0].instance_id, instances[1].instance_id)
    
    # Start continuous sync
    logger.info("\n📋 STEP 5: Starting Continuous Sync")
    sync.start_continuous_sync(sync_interval=300)  # 5 minutes
    
    logger.info("\n" + "="*60)
    logger.info("✅ Distributed Intelligence System Active")
    logger.info("Both AI instances are now sharing knowledge!")
    logger.info(f"Sync interval: 5 minutes")
    logger.info("="*60 + "\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
        sync.stop_continuous_sync()

if __name__ == "__main__":
    main()

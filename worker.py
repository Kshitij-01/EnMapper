"""
EnMapper Background Worker

Phase 0: Basic worker setup for processing background tasks.
Handles Flex and Batch processing lanes.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from settings import get_settings
from core.health import HealthChecker
from core.policy import PolicyEngine
from core.providers import ModelRegistry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnMapperWorker:
    """
    Background worker for EnMapper processing.
    
    Phase 0: Basic framework for task processing.
    Later phases will implement full queue management and task execution.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.health_checker = None
        self.policy_engine = None
        self.model_registry = None
        self.running = False
        
    async def initialize(self):
        """Initialize worker components."""
        logger.info("🔧 Initializing EnMapper Worker")
        
        try:
            # Initialize health checker
            self.health_checker = HealthChecker()
            
            # Initialize policy engine
            self.policy_engine = PolicyEngine()
            
            # Initialize model registry
            self.model_registry = ModelRegistry()
            await self.model_registry.initialize()
            
            logger.info("✅ Worker components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize worker: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform worker health check."""
        try:
            health_results = await self.health_checker.check_all()
            provider_health = await self.model_registry.get_provider_health()
            
            return {
                "worker": {
                    "status": "healthy",
                    "running": self.running,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "dependencies": health_results,
                "providers": provider_health
            }
            
        except Exception as e:
            logger.error(f"Worker health check failed: {e}")
            return {
                "worker": {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single task.
        
        Phase 0: Placeholder implementation.
        Later phases will implement full task processing logic.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")
        
        logger.info(f"🔄 Processing task {task_id} of type {task_type}")
        
        try:
            # Phase 0: Simulate task processing
            await asyncio.sleep(1.0)
            
            result = {
                "task_id": task_id,
                "type": task_type,
                "status": "completed",
                "processed_at": datetime.utcnow().isoformat(),
                "worker_id": "worker-001"
            }
            
            logger.info(f"✅ Task {task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {e}")
            return {
                "task_id": task_id,
                "type": task_type,
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat(),
                "worker_id": "worker-001"
            }
    
    async def run_worker_loop(self):
        """
        Main worker processing loop.
        
        Phase 0: Simple loop that demonstrates worker functionality.
        Later phases will implement Redis queue consumption.
        """
        logger.info("🚀 Starting worker processing loop")
        self.running = True
        
        loop_count = 0
        
        try:
            while self.running:
                loop_count += 1
                
                # Phase 0: Demonstrate worker is alive
                if loop_count % 60 == 0:  # Every 60 iterations (~ 1 minute)
                    health = await self.health_check()
                    healthy_providers = health.get("providers", {}).get("healthy_providers", 0)
                    logger.info(f"💓 Worker heartbeat - {healthy_providers} healthy providers")
                
                # Phase 0: Simulate work
                await asyncio.sleep(1.0)
                
                # TODO: Later phases will:
                # 1. Poll Redis queues for tasks
                # 2. Process LLM inference tasks
                # 3. Handle retry logic and error handling
                # 4. Update task status in database
                
        except asyncio.CancelledError:
            logger.info("🛑 Worker loop cancelled")
        except Exception as e:
            logger.error(f"❌ Worker loop failed: {e}")
        finally:
            self.running = False
            logger.info("🛑 Worker processing loop stopped")
    
    async def shutdown(self):
        """Graceful shutdown of worker."""
        logger.info("🛑 Shutting down EnMapper Worker")
        
        self.running = False
        
        # Clean up components
        if self.model_registry:
            await self.model_registry.close()
        
        if self.health_checker:
            await self.health_checker.close()
        
        logger.info("✅ Worker shutdown complete")


# Global worker instance
worker = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"📶 Received signal {signum}")
    if worker:
        asyncio.create_task(worker.shutdown())


async def main():
    """Main worker entry point."""
    global worker
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and initialize worker
        worker = EnMapperWorker()
        await worker.initialize()
        
        # Start worker loop
        await worker.run_worker_loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 Worker interrupted by user")
    except Exception as e:
        logger.error(f"❌ Worker failed: {e}")
        sys.exit(1)
    finally:
        if worker:
            await worker.shutdown()


if __name__ == "__main__":
    # Log startup
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("🚀 EnMapper Background Worker Starting")
    logger.info(f"   Environment: {settings.environment}")
    logger.info(f"   Debug: {settings.debug}")
    logger.info("=" * 60)
    
    try:
        # Run the worker
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker startup failed: {e}")
        sys.exit(1)

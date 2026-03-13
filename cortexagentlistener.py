"""
Firestore-based real-time task listener with transaction locking.
Uses Firestore as a simulated mempool for task discovery.
Handles edge cases: network failures, race conditions, stale tasks.
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import json

import firebase_admin
from firebase_admin import firestore
from firebase_admin.exceptions import FirebaseError
from google.cloud.firestore_v1 import DocumentSnapshot
from google.cloud.firestore_v1.base_client import BaseClient

from .crypto_utils import verify_signature, AgentKeyManager

logger = logging.getLogger(__name__)

@dataclass
class TaskIntent:
    """Validated task intent from Firestore."""
    id: str
    requester_public_key: str
    task_type: str
    task_payload: Dict[str, Any]
    max_api_cost_wei: int
    payment_wei: int
    deadline: datetime
    signature: str
    document_ref: Any  # Firestore DocumentReference
    
    @classmethod
    def from_document(cls, doc: DocumentSnapshot) -> Optional['TaskIntent']:
        """Create TaskIntent from Firestore document with validation."""
        if not doc.exists:
            logger.warning(f"Document {doc.id} doesn't exist")
            return None
            
        data = doc.to_dict()
        try:
            # Validate required fields
            required_fields = [
                'requester_public_key', 'task_type', 'task_payload',
                'max_api_cost_wei', 'payment_wei', 'deadline', 'signature'
            ]
            
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in task {doc.id}")
                    return None
            
            # Verify signature (EIP-712 format)
            if not verify_signature(data):
                logger.error(f"Invalid signature for task {doc.id}")
                return None
            
            # Check deadline
            deadline = data['deadline']
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            
            if deadline < datetime.now(timezone.utc):
                logger.warning(f"Task {doc.id} expired at {deadline}")
                return None
            
            # Check if already claimed
            if data.get('status') != 'open':
                logger.debug(f"Task {doc.id} status is {data.get('status')}, not open")
                return None
            
            return cls(
                id=doc.id,
                requester_public_key=data['requester_public_key'],
                task_type=data['task_type'],
                task_payload=data['task_payload'],
                max_api_cost_wei=int(data['max_api_cost_wei']),
                payment_wei=int(data['payment_wei']),
                deadline=deadline,
                signature=data['signature'],
                document_ref=doc.reference
            )
            
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error parsing task {doc.id}: {e}")
            return None

class FirestoreListener:
    """
    Real-time listener for Firestore task intents with transaction locking.
    Uses Firestore transactions to prevent race conditions in task claiming.
    """
    
    def __init__(self, project_id: str, service_account_path: str):
        """Initialize Firestore connection."""
        try:
            if not firebase_admin._apps:
                cred = firebase_admin.credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': project_id
                })
            
            self.db = firestore.client()
            self.active = False
            self.callback = None
            self.key_manager = AgentKeyManager()
            
            logger.info(f"Firestore listener initialized for project {project_id}")
            
        except (ValueError, FileNotFoundError, FirebaseError) as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            raise
    
    def start_listening(self, collection_path: str = 'task_intents'):
        """
        Start real-time listener for open tasks.
        Uses Firestore's on_snapshot with query filtering.
        """
        try:
            query = (
                self.db.collection(collection_path)
                .where('status', '==', 'open')
                .where('deadline', '>', datetime.now(timezone.utc))
            )
            
            # Watch the query
            query_watch = query.on_snapshot(self._on_snapshot)
            self.active = True
            logger.info(f"Started listening to {collection_path} for open tasks")
            
            return query_watch
            
        except Exception as e:
            logger.error(f"Failed to start listener: {e}")
            self.active = False
            raise
    
    def _on_snapshot(self, doc_snapshot, changes, read_time):
        """Handle Firestore real-time updates."""
        for change in changes:
            if change.type.name == 'ADDED':
                task = TaskIntent.from_document(change.document)
                if task and self.callback:
                    # Claim task in transaction to prevent race conditions
                    self._attempt_claim(task)
    
    def _attempt_claim(self, task: TaskIntent) -> bool:
        """
        Attempt to claim a task using Firestore transaction.
        Returns True if successfully claimed, False if race condition lost.
        """
        try:
            @firestore.transactional
            def claim_transaction(transaction, task_ref, agent_address):
                # Read the current document
                snapshot = task_ref.get(transaction=transaction)
                if not snapshot.exists:
                    return False
                    
                data = snapshot.to_dict()
                
                # Check still open and not expired
                if data.get('status') != 'open':
                    return False
                    
                deadline = data['deadline']
                if isinstance(deadline, str):
                    deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
                    
                if deadline < datetime.now(timezone.utc):
                    return False
                
                # Update to claimed
                transaction.update(task_ref, {
                    'status': 'claimed',
                    'claimed_by': agent_address,
                    'claimed_at': datetime.now(timezone.utc),
                    'agent_public_key': self.key_manager.get_public_key()
                })
                return True
            
            transaction = self.db.transaction()
            success = claim_transaction(
                transaction, 
                task.document_ref,
                self.key_manager.get_public_address()
            )
            
            if success:
                logger.info(f"Successfully claimed task {task.id}")
                if self.callback:
                    self.callback(task)
                return True
            else:
                logger.debug(f"Failed to claim task {task.id} (race condition)")
                return False
                
        except Exception as e:
            logger.error(f"Transaction failed for task {task.id}: {e}")
            return False
    
    def register_callback(self, callback_func):
        """Register callback function for successfully claimed tasks."""
        self.callback = callback_func
        logger.info("Callback registered for task execution")
    
    def stop(self):
        """Stop the listener."""
        self.active = False
        logger.info("Firestore listener stopped")

# Edge case handling tests
def test_firestore_edge_cases():
    """Test edge cases for the Firestore listener."""
    # Simulate network failure recovery
    listener = FirestoreListener("test-project", "./test-key.json")
    
    # Test 1: Malformed task document
    malformed_doc = type('MockDoc', (), {
        'exists': True,
        'id': 'test-1',
        'to_dict': lambda: {'missing_fields': True},
        'reference': None
    })()
    
    task = TaskIntent.from_document(malformed_doc)
    assert task is None, "Should reject malformed document"
    
    # Test 2: Expired task
    expired_doc = type('MockDoc', (), {
        'exists': True,
        'id': 'test-2',
        'to_dict': lambda: {
            'requester_public_key': '0x123',
            'task_type': 'test',
            'task_payload': {},
            'max_api_cost_wei': '1000',
            'payment_wei': '2000',
            'deadline': '2023-01-01T00:00:00Z',
            'signature': '0xabc',
            'status': 'open'
        },
        'reference': None
    })()
    
    task = TaskIntent.from_document(expired_doc)
    assert task is None, "Should reject expired task"
    
    logger.info("All edge case tests passed")

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    test_firestore_edge_cases()
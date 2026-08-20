import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class BufferedMessage:
    """Wraps a message with its timestamp for priority queue sorting."""
    timestamp: int  # Use nanoseconds from ROS 2
    # topic: str = field(compare=False)
    connection: Any = field(compare=False)
    data: Any = field(compare=False)

    
class JitterBuffer:
    def __init__(self, delay_ns: int = 1_000_000_000, max_latency_ns: int = 1_000_000_000):
        self.buffer = [] 
        self.delay_ns = delay_ns
        self.max_latency_ns = max_latency_ns
        self.current_time = 0
        self.camera_topics = {
            # "/oak_d_lr_1/rgb/image_raw", 
            # "/oak_d_lr_2/rgb/image_raw", 
            # "/oak_d_lr_3/rgb/image_raw"
        }

    def push(self, timestamp: int, connection: Any, data: Any):
        # RULE: Ignore messages that arrive > 1000ms late relative to stream time
        if self.current_time > 0 and (self.current_time - timestamp) > self.max_latency_ns:
            return # Drop silently or log it

        heapq.heappush(self.buffer, BufferedMessage(timestamp, connection, data))
        self.current_time = max(self.current_time, timestamp)

    def pop_ready(self):
        while self.buffer and (self.current_time - self.buffer[0].timestamp) >= self.delay_ns:
            msg = heapq.heappop(self.buffer)
            
            # RULE: Squash consecutive camera measurements
            # Peek at the next message in the heap
            if msg.connection.topic in self.camera_topics:
                while self.buffer and self.buffer[0].connection.topic == msg.connection.topic:
                    # The next message is the same camera! 
                    # Drop the current (older) one and take the next one instead
                    msg = heapq.heappop(self.buffer)
                    print(f"Squashing consecutive camera frame from topic {msg.connection.topic} at timestamp {msg.timestamp}.")
                    # We continue the while loop to handle 3+ consecutive frames
            
            yield msg

    def flush(self):
        """Yields all remaining messages (call this at end of bag)."""
        while self.buffer:
            yield heapq.heappop(self.buffer)
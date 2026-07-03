import asyncio
import websockets
import json
import base64
from pathlib import Path

class GaussianSplattingStreamer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
    
    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def send_gaussian_data(self, glb_data, frame_id):
        """Send .glb file as base64 encoded string"""
        if self.clients:
            # Encode binary .glb data to base64
            encoded_data = base64.b64encode(glb_data).decode('utf-8')
            
            message = json.dumps({
                'type': 'gaussian_splat',
                'frame_id': frame_id,
                'data': encoded_data,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Broadcast to all connected clients
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        await self.register(websocket)
    
    async def start_server(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

# Usage in your processing loop
async def main():
    streamer = GaussianSplattingStreamer()
    
    # Start server in background
    server_task = asyncio.create_task(streamer.start_server())
    
    frame_id = 0
    while True:
        # Your Gaussian Splatting processing
        glb_data = get_latest_glb_data()  # Returns binary .glb data
        
        # Stream to frontend
        await streamer.send_gaussian_data(glb_data, frame_id)
        
        frame_id += 1
        await asyncio.sleep(0.5)  # 2 FPS

if __name__ == "__main__":
    asyncio.run(main())
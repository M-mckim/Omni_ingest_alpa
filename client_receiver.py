import cv2
import socketio
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
)

# client side
socket = socketio.AsyncClient()

room_name = 'room1'

my_peer_connection = RTCPeerConnection()

@socket.on('message')
async def on_message(msg):
    print("msg : ", msg)
    socket.emit("msg_ok", room_name)


@socket.on('connect')
async def connect_on():
    print("I'm connected!")
    make_connection()
    socket.emit('join_room', room_name)
    print("join_room sent")

@socket.on('connect_error')
async def connect_error(data):
    print("The connection failed!")

@socket.on('disconnect')
async def connect_off():
    print("I'm disconnected!")
    
@socket.on('welcome')
async def on_welcome():
    print("received ywelcome")
    #offer = await m_peer_connection.createOffer()
    #await my_peer_connection.setLocalDescription(offer)
    #socket.emit("offer", offer, room_name)
    #print("sent the offer")
    
@socket.on('offer')
async def on_offer(offer):
    print("received the offer")
    print(offer)
    #await my_peer_connection.setRemoteDescription(offer)
    #answer = await my_peer_connection.createAnswer()
    #await my_peer_connection.setLocalDescription(answer)
    #socket.emit("answer", answer, room_name)
    print("sent the answer")

@socket.on('answer')
async def on_answer(answer):
    my_peer_connection.setRemoteDescription(answer)
    print("received the answer")

@socket.on('ice')
async def on_ice(ice):
    my_peer_connection.addIceCandidate(ice)
    print("received candidate")


def make_connection():
    global my_peer_connection
    my_peer_connection = RTCPeerConnection(
        configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                        "stun:stun2.l.google.com:19302",
                    ],
                },
            ],
        }
    )
    my_peer_connection.on("icecandidate", handle_ice)
    my_peer_connection.on("track", handle_add_stream)


def handle_ice(candidate):
    socket.emit("ice", candidate, room_name)
    print("sent candidate")

def handle_add_stream(track):
    print("received track")
    
@my_peer_connection.on("track")
async def on_track(track):
    print("received track : ",track)
    if track.kind == "video":
        print("received video")
        while True:
            frame = await track.recv()
            print(frame)
            # convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            # display image using cv2
            cv2.imshow("Video", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
async def main():
    await socket.connect('http://localhost:3000')
    print('my sid is', socket.get_sid())
    await socket.wait()
        
### main ###
if __name__ == '__main__':
    asyncio.run(main())

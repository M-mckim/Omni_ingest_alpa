
import socketio
import tracemalloc
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
)


tracemalloc.start()
# client side
socket = socketio.Client()

room_name = 'room1'

my_peer_connection = RTCPeerConnection()
#my_data_channel = None


@socket.on('connect')
def connect():
    print("I'm connected!")
    make_connection()
    socket.emit('join_room', room_name)
    print("join_room sent")

@socket.on('connect_error')
def connect_error(data):
    print("The connection failed!")

@socket.on('disconnect')
def disconnect():
    print("I'm disconnected!")
    
@socket.on('welcome')
async def on_welcome():
    global my_peer_connection
    #global my_data_channel
    print("received welcome")
    #my_data_channel = my_peer_connection.createDataChannel("myDataChannel")
    #my_data_channel.on("message", print)
    print("made data channel")
    offer = await my_peer_connection.createOffer()
    await my_peer_connection.setLocalDescription(offer)
    socket.emit("offer", offer, room_name)
    print("sent the offer")
    
@socket.on('offer')
async def on_offer(offer):
    global my_peer_connection
    # @my_peer_connection.on("datachannel")
    # def on_datachannel(channel):
    #     global my_data_channel
    #     my_data_channel = channel
    #     my_data_channel.on("message", lambda message: print(message.data))
    
    print("received the offer")
    await my_peer_connection.setRemoteDescription(offer)
    answer = await my_peer_connection.createAnswer()
    await my_peer_connection.setLocalDescription(answer)
    socket.emit("answer", answer, room_name)
    print("sent the answer")

@socket.on('answer')
async def on_answer(answer):
    global my_peer_connection
    await my_peer_connection.setRemoteDescription(answer)
    print("received the answer")

@socket.on('ice')
async def on_ice(ice):
    global my_peer_connection
    await my_peer_connection.addIceCandidate(ice)
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


async def handle_ice(candidate):
    socket.emit("ice", candidate, room_name)
    print("sent candidate")

async def handle_add_stream(track):
    print("received track")\
        
### main ###
socket.connect('http://localhost:3000')

print('my sid is', socket.get_sid()) # socket.sid is private sid of client (don't use it)
                                # socket.get_sid() is public sid of client

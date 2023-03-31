import cv2
import numpy as np
import asyncio
import socketio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer

sio = socketio.AsyncClient()

pc = RTCPeerConnection()

room_name = 'room1'

pcs = set()
pcs.add(pc)

@sio.event
async def connect():
    print('connection established')
    try:
        make_connection()
        print('make_connection success')
    except Exception as e:
        print('make_connection error : ',e)
    await sio.emit('join_room', 'room1')
    
@sio.event
async def message(data):
    print('message received with ', data)


@sio.event
async def disconnect():
    print('disconnected from server')
    
@sio.event   
async def welcome():
    print('received welcome')
    # await sio.emit('message', ('is it?!!!!!!!!!', room_name))
    #my_data_channel = pc.createDataChannel("datachannel")
    #print('my_data_channel : ',my_data_channel)

    offer = await pc.createOffer()
    # print('created offer : ',offer)
    
    pc.setLocalDescription(offer) # ICE Candidate 때문에 await 제거했음 문제 생기면 다시 await 추가
    # try:
    #     pc.setLocalDescription(offer) # ICE Candidate 때문에 await 제거했음 문제 생기면 다시 await 추가
    #     print('setLocalDescription(offer) success')
    # except Exception as e:
    #     print('setLocalDescription(offer) error')
    
    offer_json = {"sdp": offer.sdp, "type": offer.type}
    
    # print("offer_json : ",offer_json)
    await sio.emit("offer", (offer_json, room_name))
    print('sent the offer')
    
@sio.event
async def offer(offer):
    print('received the offer')
    # print("received offer : ", offer)
    rev_offer = RTCSessionDescription(offer['sdp'], offer['type'])
    # print("rev_ offer : ", rev_offer)
    await pc.setRemoteDescription(rev_offer)
    # try:
    #     await pc.setRemoteDescription(rev_offer)
    #     print('setRemoteDescription(offer) success!')
    # except Exception as e:
    #     print('setRemoteDescription(offer) error : ',e)
    
        
    try:
        answer = await pc.createAnswer()
        print('createAnswer success!')
    except Exception as e:
        print('createAnswer error : ',e)
    answer = await pc.createAnswer()
    # print('answer type : ',type(answer))
    # print('answer : ',answer)
    await pc.setLocalDescription(answer)  # ICE Candidate 때문에 await 제거했음 문제 생기면 다시 await 추가
    answer_json = {"sdp": answer.sdp, "type": answer.type}
    await sio.emit("answer", (answer_json, room_name))
    print('sent the answer')

@sio.on('answer')
async def on_answer(answer):
    print("received the answer")
    rev_answer = RTCSessionDescription(sdp=answer['sdp'], type=answer['type'])
    try:
        await pc.setRemoteDescription(rev_answer)  # ICE Candidate 때문에 await 제거했음 문제 생기면 다시 await 추가
        print('setRemoteDescription(answer) success!')
    except Exception as e:
        print('setRemoteDescription(answer) error : ',e)

@sio.on('ice')
async def on_ice(ice):
    if ice:
        print("received candidate")
        candidate = ice['candidate'].replace("candidate:", "")
        splitted_data = candidate.split(" ")
#rtc_candidate = RTCIceCandidate(candidate.get("component"),
#                                candidate.get("foundation"),
#                                candidate.get("address"),
#                                candidate.get("port"),
#                                candidate.get("priority"),
#                                candidate.get("protocol"),
#                                candidate.get("type"),
#                                candidate.get("relatedAddress"),
#                                candidate.get("relatedPort"),
#                                candidate.get("sdpMid"),
#                                candidate.get("sdpMLineIndex"),
#                                candidate.get("tcpType"))
        rtc_candidate = RTCIceCandidate(
                            foundation=splitted_data[0],
                            component=splitted_data[1],
                            protocol=splitted_data[2],
                            priority=int(splitted_data[3]),
                            ip=splitted_data[4],
                            port=int(splitted_data[5]),
                            type=splitted_data[7],
                            sdpMid=ice['sdpMid'],
                            sdpMLineIndex=ice['sdpMLineIndex'],
        )
        # print('rev_ice : ',rtc_candidate)
        await pc.addIceCandidate(rtc_candidate)
    
def make_connection():
    pc = RTCPeerConnection(
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
    pc.on("icecandidate", handle_ice)
    pc.on("addstream", handle_add_stream)
    
@pc.on("icecandidate")
async def handle_ice(candidate):
    print("created candidate :", candidate)
    sio.emit("ice", candidate, room_name)
    print("sent candidate")

@pc.on("addstream")
async def handle_add_stream(stream):
    print("received addstream : ", stream)

@pc.on("track")
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
    await sio.connect('http://localhost:3000')
    await sio.wait()

if __name__ == '__main__':
    asyncio.run(main())
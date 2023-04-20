import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
import numpy as np
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

from train import ImageSRTrainer
import torch
import time


ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

base_time = time.time()


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

        # Create a named window
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 480, 640)
        
        cv2.namedWindow("Raw video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Raw video", 480, 640)
        
        
    async def recv(self):
        frame = await self.track.recv()
        ftime = time.time()
        print(f"FPS: {(ftime-base_time):.6f} seconds")
        
        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")
            
            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            # Display the frame using OpenCV
            cv2.imshow("Video", img)
            cv2.waitKey(1)
            
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            # Display the frame using OpenCV
            cv2.imshow("Video", img)
            cv2.waitKey(1)
            
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            img = frame.to_ndarray(format="bgr24")
            
            cv2.imshow("Raw video", img)
            cv2.waitKey(1)
            
            # start = time.time()
            # # img super resolution
            # lr_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # lr_img = cv2.resize(lr_img,(480,270),interpolation=cv2.INTER_CUBIC) # dsize(width, height)
            # lr_img = lr_img.astype(np.float32)
            # lr_img = lr_img / 255.
            # lr_img = torch.from_numpy(np.transpose(lr_img,(2,0,1))).float()
            # lr_img = torch.unsqueeze(lr_img,0) # numpy(C x H x W) -> tensor(N x C x H x W)
            # lr_img = lr_img.to(trainer.device)

            # model.eval()
            # model.to(trainer.device)
            # sr_img = model(lr_img)

            # sr_img = torch.squeeze(sr_img,0) # tensor(N x C x H x W) -> numpy(C x H x W)
            # sr_img = trainer.T2P(sr_img)
            
            # sr_img = np.array(sr_img)
            # sr_img = cv2.cvtColor(sr_img,cv2.COLOR_RGB2BGR)
            # sr_img = cv2.resize(sr_img,(480,640),interpolation=cv2.INTER_CUBIC) # dsize(width, height)
            # end = time.time()
            # print(f"Time for SR: {(end-start):.6f} seconds")
            
            # cv2.imshow("Video", sr_img)
            # cv2.waitKey(1)
            
            # new_frame = VideoFrame.from_ndarray(sr_img, format="bgr24")
            # new_frame.pts = frame.pts
            # new_frame.time_base = frame.time_base
            return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    async def on_datachannel(channel):
        
        # create image window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 480, 640)
        
        img_buf = b''
        img_size = 0
        
        @channel.on("message")
        async def on_message(message):
            nonlocal img_buf, img_size
            # print("Message received: ", message)
            print("Type: ", type(message))
            
            if isinstance(message, str) and message.startswith("ping "):
                print("Received ping")
                channel.send("pong ")
            if isinstance(message, str) and message.startswith("size "):
                print("Received image size: ", message[5:])
                img_size = int(message[5:])
            if isinstance(message, bytes):

                if len(img_buf) < img_size:
                    img_buf += message
                    #print('img_buf: ', len(img_buf))
                if len(img_buf) == img_size:
                    print('final img_buf: ', len(img_buf))
                    print('img_buf: ', img_buf)
                    print('img_buf type: ', type(img_buf))
                    img_np = np.frombuffer(img_buf, np.uint8)
                    print('img_np: ', img_np)
                    print('img_np type: ', type(img_np))
                    print('img_np len: ', len(img_np))
                    if len(img_np) == 640*480*4:
                        img_np = img_np.reshape(640, 480, 4)
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                        img_np = cv2.resize(img_np, (480, 640))
                        
                        # tflite weight update by image
                        # Call train_w_img with your model and image
                        task1 = asyncio.create_task(trainer.train(model, img_np))
                        await task1
                                     
                                                                  
                        cv2.imshow("Image", img_np)
                        cv2.waitKey(1)
                        img_buf = b''
                        img_size = 0


    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
        
    os.makedirs("./Checkpoints/", exist_ok=True)
    print("cuda_is_available : ", torch.cuda.is_available())
    
    # Create an instance of the ImageSRTrainer class
    trainer = ImageSRTrainer(model_name='model', scale=4, num_epochs=50, batch_size=1)

    # Load your model
    model = trainer.model

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )

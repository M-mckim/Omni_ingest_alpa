// get DOM elements
var dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');
    captured_img = document.getElementById('capturedImage');
    

// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
    }

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function() {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    }, false);
    iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function() {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function() {
        signalingLog.textContent += ' -> ' + pc.signalingState;
    }, false);
    signalingLog.textContent = pc.signalingState;

    // connect audio / video
    pc.addEventListener('track', function(evt) {
        if (evt.track.kind == 'video'){
            document.getElementById('video').srcObject = evt.streams[0];
            console.log('V evt : ',evt)
            console.log('V evt.streams[0] : ',evt.streams[0])
        }
        else
            document.getElementById('audio').srcObject = evt.streams[0];
            console.log('A evt : ',evt)
            console.log('A evt.streams[0] : ',evt.streams[0])
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) { 
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;

        codec = document.getElementById('audio-codec').value;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('audio', codec, offer.sdp);
        }

        codec = document.getElementById('video-codec').value;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('video', codec, offer.sdp);
        }

        document.getElementById('offer-sdp').textContent = offer.sdp;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: document.getElementById('video-transform').value
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        document.getElementById('answer-sdp').textContent = answer.sdp;
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

function start() {
    document.getElementById('start').style.display = 'none';

    const video = document.getElementById('video')

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    pc = createPeerConnection();

    var time_start = null;

    function current_stamp() {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    }

    if (document.getElementById('use-datachannel').checked) {
        var parameters = JSON.parse(document.getElementById('datachannel-parameters').value);

        dc = pc.createDataChannel('chat', parameters);
        dc.onclose = function() {
            clearInterval(dcInterval);
            dataChannelLog.textContent += '- close\n';
        };
        dc.onopen = function() {
            dataChannelLog.textContent += '- open\n';
            
            
            document.getElementById('capturedImage').style.display = 'block';

            dcInterval = setInterval(function() {
                var message = "ping " + current_stamp()

                if (message["type"] === "msg") {
                    dataChannelLog.textContent += '> ' + message["payload"] + '\n';
                }
                console.log(message)

                dc.send(message);
            }, 1000);
            
        };
        dc.onmessage = function(evt) {
            dataChannelLog.textContent += '< ' + evt.data + '\n';

            if (evt.data.substring(0, 4) === 'pong') {
                // var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
                // dataChannelLog.textContent += ' RTT ' + elapsed_ms + ' ms\n';
                dataChannelLog.textContent += ' RTT ';

                const rawVideo = document.getElementById('Raw-video');
                canvas.id = "canvasImg";
                canvas.width = rawVideo.videoWidth;
                canvas.height = rawVideo.videoHeight;
                console.log('video.videoWidth : ',rawVideo.videoWidth)
                ctx.drawImage(rawVideo, 0, 0, canvas.width, canvas.height);
                captured_img.appendChild(canvas);

                dc.send("size "+(canvas.width*canvas.height*4));

                const CHUNK_SIZE = 16348;
                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const imgArray = new Uint8Array(imgData.data);
                console.log("imgData:",imgData)
                console.log("imgArray:",imgArray)
                const len = imgData.data.length;
                const n = len / CHUNK_SIZE | 0;

                console.log('Sending a total of ' + len + ' byte');
                // dc.send(len);

                for (let i = 0; i < n; i++) {
                    const start = i * CHUNK_SIZE;
                    const end = (i + 1) * CHUNK_SIZE;
                    console.log(start + ' - ' + (end - 1));
                    dc.send(imgArray.subarray(start, end));
                }

                if (len % CHUNK_SIZE) {
                    console.log('last ' + len % CHUNK_SIZE + ' byte');
                    dc.send(imgArray.subarray(n * CHUNK_SIZE));
                }

            }
        };
    }

    var constraints = {
        audio: document.getElementById('use-audio').checked,
        video: false
    };

    const constraints_raw = {
        audio: document.getElementById('use-audio').checked,
        video: {
            width: 640,
            height: 480,
            frameRate: 59
        }
    };

    if (document.getElementById('use-video').checked) {
        var resolution = document.getElementById('video-resolution').value;
        if (resolution) {
            resolution = resolution.split('x');
            constraints.video = {
                width: parseInt(resolution[0], 0),
                height: parseInt(resolution[1], 0),
                frameRate: 5
            };
        } else {
            constraints.video = { 
                frameRate: 5
            };
        }
    }


    if (constraints.audio || constraints.video) {
        if (constraints.video) {
            document.getElementById('media').style.display = 'block';
            document.getElementById('Raw-media').style.display = 'block';
            Raw_video = document.getElementById('Raw-video');
        }
        navigator.mediaDevices.getUserMedia(constraints_raw).then(function(stream) {
            Raw_video.srcObject = stream;
            console.log("Raw_frameRate: ", stream.getVideoTracks()[0].getSettings().frameRate)
            console.log("Raw track ID: ", stream.getVideoTracks()[0].id)
        }). catch(function(err) {
            alert('Could not acquire media: ' + err);
        });
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            console.log("frameRate: ", stream.getVideoTracks()[0].getSettings().frameRate)
            console.log("Raw track ID: ", stream.getVideoTracks()[0].id)
            stream.getTracks().forEach(function(track) {
                pc.addTrack(track, stream);
            });
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
        
    } else {
        negotiate();
    }

    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')
    
    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

# WebRTC로 스트리밍을 보내는 클라이언트



# WebRTC로 스트리밍을 보내는 클라이언트

# 빨간 원 그리는 함수
def draw_circle(frame, x, y, r, color):
    cv2.circle(frame, (x, y), r, color, -1)

draw_circle(frame, 100, 100, 50, (0, 0, 255))
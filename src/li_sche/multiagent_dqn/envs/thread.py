import threading

class Socket_Thread(threading.Thread):
    def __init__(self, name, socket, callback):
        threading.Thread.__init__(self)
        self.name = name
        self.socket = socket
        self.done = False
        self.buffer_size = 65535
        self.callback = callback

    def run(self):
        print("Starting " + self.name)
        while not self.done:
            fromaddr, flags, msg, notif = self.socket.sctp_recv(self.buffer_size)
            self.callback(msg)

        print("The thread ends! Bye!")
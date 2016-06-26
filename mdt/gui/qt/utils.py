from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MessageReceiver(QObject):

    text_message_signal = pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):
        """A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().

        It blocks until data is available, and one it has got something from the queue, it sends
        it to the "MainThread" by emitting a Qt Signal
        """
        super(MessageReceiver, self).__init__(*args, **kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            self.text_message_signal.emit(self.queue.get())

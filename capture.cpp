#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <assert.h>

using namespace std;
using namespace cv;

static bool m_enable = true;
static bool m_running = false;

static Mat frame;
static void *capture_thread(void *param)
{
	const char *uri = (const char *)param;

	VideoCapture capture( uri );

	cout << "opening video device " << uri << endl;
	bool open = capture.isOpened();
	assert(open);
	cout << "start capturing..." << endl;
	
	m_running = true;
	while ( m_enable ) {
		bool ready = capture.read(frame);
		assert(ready);
	}
	capture.release();
	m_running = false;

	return 0;
}

void startCapture( const char *uri )
{
	pthread_t th;
	pthread_create(&th, NULL, capture_thread, (void *)uri);
}

Mat &getImage()
{
	static Mat workingFrame;
	workingFrame = frame;
	return workingFrame;
}

void stopCapture()
{
	m_enable = false;
	while (m_running) {
		usleep(100000);
	}
}

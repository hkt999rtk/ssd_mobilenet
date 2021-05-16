#include <sys/time.h>

int get_current_ticks()
{
    static int started = 0;
    static struct timeval start_tv;

    if ( started == 0 ) {
        gettimeofday(&start_tv, 0);
        started = 1;
    }
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (int)((tv.tv_sec * 1000 + tv.tv_usec / 1000) -
            (start_tv.tv_sec * 1000 + start_tv.tv_usec / 1000));
}

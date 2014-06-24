#ifndef singleeyefitter_timing_h__
#define singleeyefitter_timing_h__

namespace singleeyefitter {

	namespace timing {

#ifdef WIN32

#include <windows.h>

		struct timer {
			LARGE_INTEGER start_time;

			timer()  {
				QueryPerformanceCounter(&start_time);
			}

			void start()  {
				QueryPerformanceCounter(&start_time);
			}

			double get_ms() {
				LARGE_INTEGER end_time;
				QueryPerformanceCounter(&end_time);
				LARGE_INTEGER freq;
				QueryPerformanceFrequency(&freq);
				return ((end_time.QuadPart - start_time.QuadPart) * 1000.0 / freq.QuadPart);
			}
		};

#else

#error Timing not supported on non win

#endif

	}

}

#endif // singleeyefitter_timing_h__

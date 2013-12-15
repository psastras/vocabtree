#pragma once

#include <string>
#include "logger.hpp"
#include "cycletimer.hpp"
#include <boost/current_function.hpp> 
#include <config.hpp>
#include <stdint.h>

#define SCOPED_TIMER \
	ScopedTimer __s(BOOST_CURRENT_FUNCTION);

class PerfTracker {
	public:
		static PerfTracker &instance();
		void add_time(const std::string &func, double time);
		std::map<std::string, std::pair<double, uint64_t> > &times();
	private:
		PerfTracker();
		~PerfTracker();

		static PerfTracker *s_instance;
		std::map<std::string, std::pair<double, uint64_t> > _times;
};

class ScopedTimer {
public:
  ScopedTimer(const std::string &f) : func(f), st(CycleTimer::currentSeconds()) { }
  ~ScopedTimer() {
  	PerfTracker::instance().add_time(func, CycleTimer::currentSeconds() - st);
  }

  std::string func;
  double st;
};

std::ostream& operator<< (std::ostream &out, PerfTracker &pt);


namespace misc {
	std::string get_machine_name();
};

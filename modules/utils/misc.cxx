#include "misc.hpp"
#include <cstring>
#ifdef WIN32
    #include <Windows.h>
    #include <tchar.h>
#else
    #include <unistd.h>
#endif

namespace misc {
    std::string get_machine_name() {
        const int max_len = 150;
        char name[max_len];
        memset(name, 0, max_len);

#ifdef WIN32
        TCHAR infoBuf[max_len];
        DWORD bufCharCount = max_len;
        
        if(GetComputerName(infoBuf, &bufCharCount)) {
            for(int i=0; i<max_len; i++) {
                name[i] = infoBuf[i];
            }
        }
        else {
            strcpy(name, "Unknown_Host_Name");
        }
#else
        gethostname(name, max_len);
#endif
        std::string machine_name = name;
        return machine_name;
    }
}

PerfTracker* PerfTracker::s_instance = 0;

PerfTracker::PerfTracker() {

}

PerfTracker::~PerfTracker() {
    
}

PerfTracker &PerfTracker::instance() {
    if(!s_instance) s_instance = new PerfTracker();
    return *s_instance;
}

void PerfTracker::add_time(const std::string &func, double time) {
    #pragma omp critical
    {
        _times[func].first += time;
        _times[func].second++;
    }
}

bool PerfTracker::save(const std::string &file_path) {
  std::ofstream ofs(file_path.c_str(), std::ios::trunc);

  ofs << *s_instance;

  return (ofs.rdstate() & std::ofstream::failbit) == 0;
}

std::map<std::string, std::pair<double, uint64_t> > &PerfTracker::times() {
    return _times;
}

std::ostream& operator<< (std::ostream &out, PerfTracker &pt) {
  out << "Performance Report" << std::endl;
  out << "Function Name\tTotal Time\tNumber of Calls\tAverage Time" << std::endl;
  out << "==================================================" << std::endl;
  for(std::map<std::string, std::pair<double, uint64_t> >::iterator it = pt.times().begin(); it != pt.times().end(); ++it) {
    out << it->first << "\t" << it->second.first << "\t" <<
     it->second.second << "\t" << it->second.first / (double)it->second.second << std::endl;
  }
  return out;
}
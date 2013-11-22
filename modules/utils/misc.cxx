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
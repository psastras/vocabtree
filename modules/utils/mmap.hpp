/// This file provides a cross platform way of accessing mmap
#pragma once
#if !defined(WIN32) && defined(UNIX)
#include <sys/mman.h>
#elif defined(WIN32) && !defined(UNIX)

// mmap prot flags
#define PROT_NONE   0x00	// no access (not supported on Win32)
#define PROT_READ	0x01
#define PROT_WRITE	0x02

// mmap flags
#define MAP_SHARED	0x01	// share changes across processes
#define MAP_PRIVATE	0x02	// private copy-on-write mapping
#define MAP_FIXED	0x04

#define MAP_FAILED 0

extern void* mmap(void* start, size_t len, int prot, int flags, int fd, off_t offset);
extern int munmap(void* start, size_t len);



void* mmap(void* const user_start, const size_t len, const int prot, const int flags, const int fd, const off_t offset) {
	{
		WIN_SAVE_LAST_ERROR;

		// assume fd = -1 (requesting mapping backed by page file),
		// so that we notice invalid file handles below.
		HANDLE hFile = INVALID_HANDLE_VALUE;
		if(fd != -1) {
			hFile = mk_handle(_get_osfhandle(fd));
			if(hFile == INVALID_HANDLE_VALUE) {
				debug_warn("mmap: invalid file handle");
				goto fail;
			}
		}

		// MapView.. will choose start address unless MAP_FIXED was specified.
		void* start = 0;
		if(flags & MAP_FIXED) {
			start = user_start;
			if(start == 0)	// assert below would fire
				goto fail;
		}

		// figure out access rights.
		// note: reads are always allowed (Win32 limitation).

		SECURITY_ATTRIBUTES sec = { sizeof(SECURITY_ATTRIBUTES), (void*)0, FALSE };
		DWORD flProtect = PAGE_READONLY;
		DWORD dwAccess = FILE_MAP_READ;

		// .. no access: not possible on Win32.
		if(prot == PROT_NONE)
			goto fail;
		// .. write or read/write (Win32 doesn't support write-only)
		if(prot & PROT_WRITE) {
			flProtect = PAGE_READWRITE;

			const bool shared = (flags & MAP_SHARED ) != 0;
			const bool priv   = (flags & MAP_PRIVATE) != 0;
			// .. both aren't allowed
			if(shared && priv)
				goto fail;
			// .. changes are shared & written to file
			else if(shared)
			{
				sec.bInheritHandle = TRUE;
				dwAccess = FILE_MAP_ALL_ACCESS;
			}
			// .. private copy-on-write mapping
			else if(priv)
			{
				flProtect = PAGE_WRITECOPY;
				dwAccess = FILE_MAP_COPY;
			}
		}

		// now actually map.
		const DWORD len_hi = (DWORD)((u64)len >> 32);
			// careful! language doesn't allow shifting 32-bit types by 32 bits.
		const DWORD len_lo = (DWORD)len & 0xffffffff;
		const HANDLE hMap = CreateFileMapping(hFile, &sec, flProtect, len_hi, len_lo, (LPCSTR)0);
		if(hMap == INVALID_HANDLE_VALUE)
			// bail now so that MapView.. doesn't overwrite the last error value.
			goto fail;
		void* ptr = MapViewOfFileEx(hMap, dwAccess, len_hi, offset, len_lo, start);

		// free the mapping object now, so that we don't have to hold on to its
		// handle until munmap(). it's not actually released yet due to the
		// reference held by MapViewOfFileEx (if it succeeded).
		if(hMap != INVALID_HANDLE_VALUE)	// avoid "invalid handle" error
			CloseHandle(hMap);

		if(!ptr)
			// bail now, before the last error value is restored,
			// but after freeing the mapping object.
			goto fail;

		assert(!(flags & MAP_FIXED) || (ptr == start));
			// fixed => ptr = start

		WIN_RESTORE_LAST_ERROR;

		return ptr;
	}
fail:
	return MAP_FAILED;
}


int munmap(void* const start, const size_t len) {
	UNUSED(len);
	BOOL ok = UnmapViewOfFile(start);
	return ok? 0 : -1;
}

#endif
#!/usr/bin/python

from config import *;
import os;
import sys;

if __name__ == "__main__":
	print BENCH_FEATURES_EXE;
	os.system(BENCH_FEATURES_EXE);
	print BENCH_BOW_EXE;
	os.system(BENCH_BOW_EXE)
	print BENCH_INDEX_EXE;
	os.system(BENCH_INDEX_EXE)
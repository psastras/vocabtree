#!/usr/bin/python

from config import *;
import os;
import sys;

if __name__ == "__main__":
	print TESTS_FEATURES_EXE;
	os.system(TESTS_FEATURES_EXE);
	print TESTS_BOW_EXE;
	os.system(TESTS_BOW_EXE)
	print TESTS_INDEX_EXE;
	os.system(TESTS_INDEX_EXE)
	print TESTS_SEARCH_EXE;
	os.system(TESTS_SEARCH_EXE)
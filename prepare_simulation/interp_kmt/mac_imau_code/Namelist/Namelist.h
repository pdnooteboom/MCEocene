/****************************** Namelist.h *****************************
**
** $Version$
** $Date: 2004/02/02 04:28:30 $
** $Source: /usr/local/cvsroot/HYPOP++/Namelist/Namelist.h,v $
** $Author: bob $
**
*/

#ifndef NamelistH
#define NamelistH

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "Item.h"

/**************************** class Namelist *******************************/

class Namelist
{
  	string name_;
	Nl_error_number nl_error_level_;
	vector<Item> entries_;

	/***** private member error routines *****/
	
	void error_file_missing(string file_name);
	void error_open_file(string file_name);
	void error_namelist_missing(string file_name);
	void error_no_match(string file_name, string var_name);
	void error_meaningless_line(string file_name, char* line);
	void error_string_length(string type);
	
	
  public:
 	Namelist() 
	{ };

 	Namelist(const char*  n, bool all_fatal = true);
	
	Namelist(const string n, bool all_fatal = true); 
	
	void entry(const char*  n, int* ip, int nv = 1);
	void entry(const char*  n, long* ptr, int nv = 1);
	void entry(const char*  n, float* ptr, int nv = 1);
	void entry(const char*  n, double* ptr, int nv = 1);
	void entry(const char*  n, bool* ptr, int nv = 1);
	void entry(const char*  n, char* ptr, int nv = 1);
	void entry(const char*  n, string* ptr, int nv = 1);
	
	string	name() { return name_; }
	int		entries() { return entries_.size(); }
 	void	read(char* file_name);
	void	read(string file_name);	// { read(file_name.c_str()); }
	void	print(char* comment);		// prints namelist variables and values
};

#endif 

/*
** $Log: Namelist.h,v $
** Revision 1.4  2004/02/02 04:28:30  bob
** Made some corrections to the new nl_error routines and calls to them
**
** Revision 1.3  2004/02/01 20:12:09  bob
** In the process of moving Namelist error message to Errors.cpp
**
*/

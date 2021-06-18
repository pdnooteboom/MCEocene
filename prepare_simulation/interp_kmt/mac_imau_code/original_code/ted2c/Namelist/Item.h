/****************************** Item.h *****************************
**
** $Version$
** $Date: 2004/02/01 20:12:09 $
** $Source: /usr/local/cvsroot/HYPOP++/Namelist/Item.h,v $
** $Author: bob $
**
*/

#ifndef ItemH
#define ItemH

#include <string>
using namespace std;

class Item
{
  	string	name_;		// name of variable to be updated
	string	type_;		// type of variable to be updated
 	void*	ptr_;		// address of variable to be updated
	int		nvalues_;	// if nvalues_ > 1, ptr_ is the first element
						// of a 1-D array ptr[nvalues] for which storage
						// has been allocated in the calling routine.
						
	void error_convert_long(string s, long l);
	void error_convert_double(string s, double& d);
	void error_update_value(string type);
						
  public:
 	Item() { };
	
	// pass by value so that Item will make a copy of its own
	Item(const string n, const string t, void* p, int nv = 1)
	{
		name_ = n;
		type_ = t;
		ptr_ = p;
		nvalues_ = nv;
	};
 	
	~Item() { };		// destructor
	
	Item& operator =(const Item& e);
	
 	string name() { return name_; }
 	string type() { return type_; }
 	void* ptr()  { return ptr_; }
	int nvalues() { return nvalues_; }

	void update_value(char* value, int iv);
};


/*********************** Namelist Errors ********************************

Defines a set of error codes that specify the seriousness of the error
and which control, with user input, the response to the error. At 
present, errors are grouped into three categories:
		Always (unconditionally) fatal -- no user response possible
		Serious -- if user chooses not to respond, error becomes fatal.
		Warning -- code ignores error and continues unless user specifies
				   not by lowering the default value of nl_error_level.
*/


enum Nl_error_number 
{  
	nl_error_none				= 0,	// No error
	
	// Warning errors					// USED IN		IN FILE
	nl_error_warning			= 10,
	nl_error_file_missing		= 11,	// read			Namelist.cpp
	nl_error_namelist_missing	= 12,	// read			Namelist.cpp
	nl_error_no_match			= 13,	// read			Namelist.cpp
	nl_error_meaningless_line	= 14,	// read			Namelist.cpp
	nl_error_open_file			= 15,	// read			Namelist.cpp
	
	// Serious errors
	nl_error_serious			= 20,
	nl_error_convert_long		= 21,	// update_value	Item.cpp
	nl_error_convert_double		= 22, 	// update_value	Item.cpp
	
	// Fatal errors
	nl_error_fatal				= 30,
	nl_error_string_length		= 31,	// update_value	Item.cpp
	nl_error_update_value		= 32,	// update_value	Item.cpp
	nl_error_out_of_space		= 33	// update_value	Item.cpp
};


#endif ItemH

/*
** $Log: Item.h,v $
** Revision 1.2  2004/02/01 20:12:09  bob
** In the process of moving Namelist error message to Errors.cpp
**
*/
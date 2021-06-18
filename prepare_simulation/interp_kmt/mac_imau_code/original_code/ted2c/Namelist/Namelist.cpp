/****************************** Namelist.cpp *****************************
**
** $Version$
** $Date: 2004/02/01 20:12:09 $
** $Source: /usr/local/cvsroot/HYPOP++/Namelist/Namelist.cpp,v $
** $Author: bob $
**
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cerrno>
using namespace std;

#include "Namelist.h"
#include "../Utilities/Utilities.h"

// extern Nl_error_number nl_error_level;

Namelist::Namelist(const char*  n, bool all_fatal) 
{ 	name_.assign(n); 
	nl_error_level_ = (all_fatal) ? nl_error_none : nl_error_fatal;
	cout << "\n\nNamelist '" << name_ << "'";
	if (all_fatal)	 
		cout << " [all errors fatal] ***********************\n";
	else
		cout << " ******************************************\n";	
}

Namelist::Namelist(const string n, bool all_fatal) 
{ 	name_ = n; 
	nl_error_level_ = (all_fatal) ? nl_error_none : nl_error_fatal;
	cout << "\n\nNamelist '" << name_ << "'";
	if (all_fatal)	 
		cout << " [all errors fatal] ***********************\n";
	else
		cout << " ******************************************\n";	
}


/******* Namelist.entry (overloaded for various types) **********/

void Namelist::entry(const char* n, int* ip, int nv)
{
	
	string type("int");
	void* vp = static_cast<void*>(ip);
	Item* item = new Item(string(n), type, vp, nv);
	entries_.push_back(*item);
}

void Namelist::entry(const char* n, long* lp, int nv)
{
	string type("long");
	void* vp = static_cast<void*>(lp);
	Item* item = new Item(string(n), type, vp, nv);
	entries_.push_back(*item);
}

void Namelist::entry(const char* n, float* fp, int nv)
{
	string type("float");
	void* vp = static_cast<void*>(fp);
	Item* item = new Item(string(n), type, vp, nv);
	entries_.push_back(*item);
}

void Namelist::entry(const char* n, double* dp, int nv)
{
	string type("double");
	void* vp = static_cast<void*>(dp);
	Item* item = new Item(string(n), type, vp, nv);
	entries_.push_back(*item);
}

void Namelist::entry(const char* n, bool* bp, int nv)
{
	string type("bool");
	void* vp = static_cast<void*>(bp);
	Item* item = new Item(string(n), type, vp, nv);
 	entries_.push_back(*item);
}

void Namelist::entry(const char* n, char* cp, int nv)
{
	string type("char");
	void* vp = static_cast<void*>(cp);
	Item* item = new Item(string(n), type, vp, nv);
 	entries_.push_back(*item);
}

void Namelist::entry(const char* n, string* sp, int nv)
{
	string type("string");
	void* vp = static_cast<void*>(sp);	
	Item* item = new Item(string(n), type, vp, nv);
 	entries_.push_back(*item);
}

/********************* Namelist Errors ***********************/
	
void Namelist::
error_file_missing(string file_name)
{
	cout << "\nWARNING: Could not open file '" << file_name
		 << "' to read Namelist '" << name_ << "' input\n";
	if (nl_error_file_missing > nl_error_level_) exit(1);
}


void Namelist::
error_open_file(string file_name)
{
	cout << "\nWARNING: Unable to open input file '" << file_name << "'"
		 << "\nWill use defaults of namelist '" << name_ << "'\n";
	if (nl_error_open_file > nl_error_level_) exit(1);
}

void Namelist::
error_namelist_missing(string file_name)
{
 	cout << "\nWARNING: Did not find Namelist '" << name_ 
		 << "' in file '" << file_name << "'\n";
	if (nl_error_namelist_missing > nl_error_level_) exit(1); 
}

void Namelist::
error_no_match(string file_name, string var_name)
{
	cout << "\nWARNING: Variable name '" << var_name 
		 << "' appears in Namelist '" << name_ << "' in file '" << file_name
		 << "'\nbut does not match any variable in code Namelist. "
		 << "Check spelling.\n";
	if (nl_error_no_match > nl_error_level_) exit(1); 
}

void Namelist::
error_meaningless_line(string file_name, char* line)
{
	cout << "\nWARNING: Cannot interpret line reading namelist '" << name_	
		 << "' in file '" << file_name << "'. \nThe line is '" << line << "'\n";
	if (nl_error_meaningless_line > nl_error_level_) exit(1);
}

// Fatal errors
void Namelist::
error_string_length(string type)
{
	cout << "\nFATAL ERROR in Item::update_value:"
		 << "\nUnexpected variable type '" << type << "'\n";
	exit(1);	 
}
		

/************************* Namelist Read ***************************

Read namelist with the following assumptions:
	 (1) the namelist begins with string start = '&' + name
	 (2) this is followed by one entry per line of the form:
 		 in_name = in_values
	 (3) the namelist input is terminanted by a line containing '/' only
*/

void Namelist::
read(string file_name)
{	
	const streamsize line_length = 255;
	char c, line[line_length];
	char *in_name, *in_value;
	char *loc_equal, *loc_comma;

//	int nl_error_number = nl_error_none;

	ifstream input;			// create link to input stream
	
	input.open(file_name.c_str());	// open it for reading
	if (input.fail()) error_open_file(file_name);
	
	// Scan input file for first line of namelist
	char* values;
	int nchars;
	bool found_namelist = false;
	while (! input.eof())
	{	input.getline(line, line_length);

		eliminate_whitespace(line);
		if (line[0]=='\0') continue;	// skip blank lines
		
		if (found_namelist)		// have already begun reading this namelist
		{	
			// check for end of namelist
			if (strncmp(line,"/",1)==0) // end of this namelist
			{	
				break;					// namelist finished, exit while loop
			}
			else if (strchr(line,'='))	// line contains '='
			{
				in_name = strtok(line,"=");
				bool matched_in_name = false;
				int ne;					// scope persists after for-loop
				for (ne = 0; ne < entries_.size(); ne++)
				{	if (entries_[ne].name() == string(in_name))	// got a match 
					{
						matched_in_name = true; 

						// see how many values this entry should have
						int nvalues = entries_[ne].nvalues();
						int nv = 0;
									
						// extract value(s) that follows '='
						in_value = strtok(0,",");
						do 
						{	entries_[ne].update_value(in_value, nv);
							in_value = strtok(0,",");
						} while (in_value && (++nv < nvalues));	
						if (! in_value) 
							break;	// no more values, leave loop
					}
				} 

				if ( ! matched_in_name ) error_no_match(file_name, in_name);
			}
			else	// line is neither '/' or of form 'name = value' 
			{	
				error_meaningless_line(file_name, line);
			}
		} // have not found namelist yet, so continue looking for it
		else if ((strncmp(line,"&",1)==0) &&		// start of some namelist
				 (strcmp(line+1,name_.c_str())==0))	// ... of requested namelist
		{	
			found_namelist = true;			// now we are reading namelist input
		}
	}

	if (! found_namelist)	// requested Namelist does not appear in input file
	{	
		error_namelist_missing(file_name);
	}

	input.close();
}
/************************* Namelist Print ***************************/ 

void Namelist::
print(char* s)		// "s" is "{initial,final} values"
{
	cout << "\nNamelist '" << name_ << "', " << entries_.size() << " entries, "<< s << endl;
	for (int i = 0; i < entries_.size(); i++)
	{	
		int nvalues = entries_[i].nvalues();

		cout << entries_[i].type() << "\t" << entries_[i].name();
		if (nvalues > 1) cout << "[" << nvalues << "]";		
		cout << "\t = ";

		for (int iv = 0; iv < nvalues; iv++)
		{
			if (entries_[i].type() == "long")
			{	
				long *p = reinterpret_cast<long*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			else if (entries_[i].type() == "int")
			{	
				int* p = reinterpret_cast<int*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			else if (entries_[i].type() == "double")
			{	
				double* p = reinterpret_cast<double*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			else if (entries_[i].type() == "float")
			{	
				float* p = reinterpret_cast<float*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			else if (entries_[i].type() == "bool")
			{	
				bool* p = reinterpret_cast<bool*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			else if (entries_[i].type() == "char")
			{	
				char* p = reinterpret_cast<char*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			else if (entries_[i].type() == "string")
			{	
				string* p = reinterpret_cast<string*>(entries_[i].ptr());
				cout << *(p+iv);
			}
			
			if (iv < nvalues-1) 
				cout << ", ";
			else
				cout << endl;
				
		} // while
	}
}


/*
** $Log: Namelist.cpp,v $
s** Revision 1.2  2004/02/01 20:12:09  bob
** In the process of moving Namelist error message to Errors.cpp
**
*/

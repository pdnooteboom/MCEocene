##### Utilities make file #####
## 
## $Revision: 1.1 $
## $Date: 2004/02/01 20:04:32 $
## $Source: /usr/local/cvsroot/HYPOP++/Utilities/makefile,v $
## $Author: bob $
##

libUtilities.a: Utilities.o Digits.o
	libtool -static -o libUtilities.a Utilities.o Digits.o

##	mv libUtilities.a ../Library

Test_eliminate_whitespace:	Utilities.o Test_eliminate_whitespace.o
	g++ -g -o Test_eliminate_whitespace Utilities.o Test_eliminate_whitespace.o

Utilities.o:	Utilities.h Utilities.cpp
	g++ -g -c Utilities.cpp

Digits.o:	Digits.h Digits.cpp
	g++ -g -c Digits.cpp
	
Test_eliminate_whitespace.o:	Utilities.h Test_eliminate_whitespace.cpp
	g++ -g -c Test_eliminate_whitespace.cpp

clean:
	rm *.o
	mv *.cpp~ ../Unneeded
	mv *.h~ ../Unneeded


##
## $Log: makefile,v $
## Revision 1.1  2004/02/01 20:04:32  bob
## Addes Utility routines, which are non-member functions of general applicability
##
##

##### Namelist make file #####
## 
## $Revision: 1.5 $
## $Date: 2004/02/02 05:57:33 $
## $Source: /usr/local/cvsroot/HYPOP++/Namelist/makefile,v $
## $Author: bob $
##

libNamelist.a: Item.o Namelist.o 
	libtool -static -o libNamelist.a Item.o Namelist.o 
#	mv libNamelist.a ../Library

TestNamelist:	Item.o Namelist.o TestNamelist.o ../Library/libUtilities.a
	g++ -g -o TestNamelist TestNamelist.o -L../Library -lNamelist -lUtilities


Item.o:	Item.h Item.cpp
	g++ -g -c Item.cpp
	
Namelist.h:	Item.h

Namelist.o:	Namelist.h Namelist.cpp
	g++ -g -c Namelist.cpp
	
TestNamelist.o:	Namelist.h TestNamelist.cpp
	g++ -g -c TestNamelist.cpp

clean:
	rm *.o
	mv *.cpp~ ../Unneeded
	mv *.h~ ../Unneeded


##
## $Log: makefile,v $
## Revision 1.5  2004/02/02 05:57:33  bob
## Minor  bug fixes in Namelist
##
## Revision 1.4  2004/02/02 04:28:30  bob
## Made some corrections to the new nl_error routines and calls to them
##
## Revision 1.3  2004/02/01 20:12:09  bob
## In the process of moving Namelist error message to Errors.cpp
##
##

##################### TED makefile ########################
##
## $Revision$
## $Date$
## $Source$
## $Author$
##


CC = g++
DEBUG = # -g
OBJS = main.o TopoData.o 

NMLLIB = -L./Namelist -lNamelist
NMLINC = -I./Namelist
GLFRWKS = -framework GLUT -framework OpenGL
UTILIB  = -L./Utilities -lUtilities

LIBS = $(NMLLIB) $(UTILIB) $(GLFRWKS) -L/usr/lib -lobjc -lm

iTed: $(OBJS) 
	$(CC) $(DEBUG ) -o iTed  $(OBJS) $(LIBS)

TopoData.o:	TopoData.h TopoData.cpp
	$(CC) $(DEBUG) -c TopoData.cpp   
	
main.o: TopoData.h Namelist/Namelist.h main.cpp
	$(CC) $(DEBUG) $(NMLINC) $(GLFRWKS) -c main.cpp


##### make all #####

clean:
	rm -f *.o
	rm -f iTed

cleanall:
	rm -f *.o iTed
	
	rm -f Namelist/*.o 
	rm -f Namelist/*.a
	rm -f Namelist/TestNamelist
	
	rm -f Utilities/*.o 
	rm -f Utilities/*.a

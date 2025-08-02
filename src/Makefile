


CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -O3
OBJS = symnmf.o

all: symnmf

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c -o symnmf.o


symnmf: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o symnmf -lm

clean:
	rm -f *.o symnmf
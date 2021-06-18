#include <iostream>
#include <cstdlib>
using namespace std;

#include "Digits.h"


/*************************************************************************/	
double  array_average(int n, double *v)
/*************************************************************************/	
{
	double sum = 0.0;
	for (int i = 0; i < n; ++i)
		sum += v[i];
	return (sum/n);
}

/*************************************************************************/	
double array_product(int n, double *g, double *h)
/*************************************************************************/	
{
	double sum = 0.0;
	for (int i = 0; i < n; ++i)
		sum += g[i] * h[i];
	return sum;
}

/*************************************************************************/	
int remove_leading_digit(int number)
/*************************************************************************/	
{
	// Count digits in 'number'
	if (number < 1) return -999999;		// operates only on positive longegers
	
	int num = number, ndigits = 0, marker = 1;
	while (num > 0)
	{
		++ndigits;
		num /= 10;
		marker *= 10;
	}
	if (ndigits > 9) cout << "\n\nremove_leading_digit: WARNING - INTEGER VALUE"
						  << " EXCEEDS 10^9, THE LIMIT OF 32-BIT ARITHMETIC\n";
	marker /= 10;
	return number % marker;
}

/*************************************************************************/	
int count_digits(int number)
/*************************************************************************/	
{
	// Count digits in 'number'
	int num = number, ndigits = 0;
	
	while (num > 0)
	{
		++ndigits;
		num /= 10;
	}
	return ndigits;	
}


/*************************************************************************/	
int digit_in_column(int c, int number)
/*************************************************************************/	
{
	// This routine extracts the digit in the column specified by c.
	// Let number be the string of digits 'ijklmn'. If c < 0, the 
	// columns are counted from the right side. Thus c = -2 would 
	// return 'm', c = -4 would return 'k'. If c > 0, the columns 
	// are counted from the left side.  Thus, if c = +3, 'k' would
	// be returned. Normally, an unsigned digit is returned. However,
	// if c = 0, the return value is -1,or if |c| > the number of 
	// digits in 'number', -2 is returned.
	
	// Count digits in 'number'
	int num = number, ndigits = 0;
	if (c == 0) return -1;

	// count the digits in 'number'
	while (num > 0)
	{
		++ndigits;
		num /= 10;
	}
	if (ndigits > 9) cout << "\n\ndigit_in_column: WARNING -- INTEGER VALUE"
						  << " EXCEEDS 10^9, THE LIMIT OF 32-BIT ARITHMETIC\n";
	if (abs(c) > ndigits) return -2;
	
	num = number;
	if (c > 0) c = ndigits - c + 1;  // convert to equivalent number from right
	if (c < 0) c = -c;
	for (int n = 1; n < c; ++n)
		num /= 10;
	// desired digit should be in rightmost column
	return (num%10);
}
#ifdef TEST
int main(int argc, char * const argv[])
{
	int entry, result,num, number = 123456789L;
	
	// First test "remove_leading_digit"
	num = number;
	while (num > 0)
	{
		cout << "\nremove_leading_digit from '" << num << "' yields '"
			 << (num = remove_leading_digit(num)) << "'\n";
	}

	while (1)
	{
		cout << "\nEnter an integer value to indicate the position to select:" 
			 << "\nA positive value counts digits relative to the left end"
			 << "\nA negative value counts digits relative to the right end"
			 << "\n\nnumber = " << number << "\nentry = ";
		cin >>  entry;
		result = digit_in_column(entry, number);
		cout << "\nThe requested digit is: " << result << endl;		 
	}
}
#endif

#pragma once

#include <stdio.h>
#include <string>
#include <stdint.h>
#include <assert.h>
#include <tchar.h>
#include <string.h>

#define XP_MIN(a, b)					((a) <= (b)?(a):(b))
#define XP_MAX(a, b)					((a) >= (b)?(a):(b))

#ifdef _WIN32
#define XP_STRLEN(s)					strlen(s)
#define XP_STRICMP						_stricmp
#define XP_STRCMP						strcmp
#define XP_STRNICMP						_strnicmp
#if defined(_MSC_VER) && _MSC_VER >= 1400
#define XP_STRCPY(s,l,ss)				strcpy_s(s,l,ss)
#define XP_STRNCPY(s,l,ss,n)			strncpy_s(s,l,ss,n)
#define XP_STRCAT(s,l,ss)				strcat_s(s,l,ss)
#define XP_STPRINTF_S(s,l,f,...)		sprintf_s(s,l,f,__VA_ARGS__)
#define	XP_SNTPRINTF_S(s,l,cc,f,...)	snprintf_s(s, l, cc, f, __VA_ARGS__)
#else
#define XP_STRCPY(s,l,ss)				strcpy(s,ss)
#define XP_STRNCPY(s,l,ss,n)			strncpy(s,ss,n)
#define XP_STRCAT(s,l,ss)				strcat(s,ss)
#define XP_STPRINTF_S(s,l,f,...)		sprintf(s,f,__VA_ARGS__)
#define	XP_SNTPRINTF_S(s,l,cc,f,...)	snprintf(s, cc, f, __VA_ARGS__)
#endif //defined(_MSC_VER) && _MSC_VER >= 1400
#elif defined(__linux__)
#define XP_STRLEN(s)					strlen(s)
#define XP_STRCMP						strcmp
#define XP_STRICMP						strcasecmp
#define XP_STRNICMP						strncasecmp
#define XP_STRCPY(s,l,ss)				strcpy(s,ss)
#define XP_STRNCPY(s,l,ss,n)			strncpy(s,ss,n)
#define XP_STRCAT(s,l,ss)				strcat(s,ss)
#define XP_STPRINTF_S(s,l,f,...)		sprintf(s,f,##__VA_ARGS__)
#define	XP_SNTPRINTF_S(s,l,cc,f,...)	snprintf(s, cc, f, ##__VA_ARGS__)
#endif // _WIN32

#if (defined(__linux__))&&(!defined(__int64))
#define __int64 long long
#endif

#ifdef _BIG_ENDIAN_
#define ENDIANUSHORT(src)           (unsigned   short)src
#define ENDIANULONG(src)			(unsigned    long)src
#define ENDIANUINT64(src)			(unsigned __int64)src
#else
#define ENDIANUSHORT(src)			((unsigned   short)((((src)>>8)&0xff) |\
														(((src)<<8)&0xff00)))

#define ENDIANULONG(src)			((unsigned    long)((((src)>>24)&0xFF) |\
														(((src)>> 8)&0xFF00) |\
														(((src)<< 8)&0xFF0000) |\
														(((src)<<24)&0xFF000000)))

#define ENDIANUINT64(src)			((unsigned __int64)((((src)>>56)&0xFF) |\
														(((src)>>40)&0xFF00) |\
														(((src)>>24)&0xFF0000) |\
														(((src)>> 8)&0xFF000000) |\
														(((src)<< 8)&0xFF00000000LL) |\
														(((src)<<24)&0xFF0000000000LL) |\
														(((src)<<40)&0xFF000000000000LL) |\
														(((src)<<56)&0xFF00000000000000LL)))
#endif //_BIG_ENDIAN_

#define DECLARE_ENDIAN_BEGIN()		void Endian(bool bBig2SysByteOrder=true){UNREFERENCED_PARAMETER(bBig2SysByteOrder);
#define DECLARE_ENDIAN_END()		return;}

#ifdef _BIG_ENDIAN_
#define USHORT_FIELD_ENDIAN(field)
#define ULONG_FIELD_ENDIAN(field)
#define UINT64_FIELD_ENDIAN(field)
#define UTYPE_FIELD_ENDIAN(field)   field.Endian();
#else
#define USHORT_FIELD_ENDIAN(field)	field = ENDIANUSHORT(((unsigned short)field));
#define ULONG_FIELD_ENDIAN(field)	field = ENDIANULONG(((unsigned long)field));
#define UINT64_FIELD_ENDIAN(field)	field = ENDIANUINT64(((unsigned long long)field));
#define UTYPE_FIELD_ENDIAN(field)	field.Endian();
#endif //_BIG_ENDIAN_

enum FLAG_VALUE
{
	FLAG_UNSET = 0,
	FLAG_SET = 1,
	FLAG_UNKNOWN = 2,
};

enum INT_VALUE_LITERAL_FORMAT
{
	FMT_AUTO = 0,
	FMT_HEX,
	FMT_DEC,
	FMT_OCT,
	FMT_BIN
};

inline bool ConvertToInt(char* ps, char* pe, int64_t& ret_val, INT_VALUE_LITERAL_FORMAT literal_fmt = FMT_AUTO)
{
	ret_val = 0;
	bool bNegative = false;

	if (pe <= ps)
		return false;

	if (literal_fmt == FMT_AUTO)
	{
		if (pe > ps + 2 && XP_STRNICMP(ps, "0x", 2) == 0)	// hex value
		{
			ps += 2;
			literal_fmt = FMT_HEX;
		}
		else if (*(pe - 1) == 'h' || *(pe - 1) == 'H')
		{
			pe--;
			literal_fmt = FMT_HEX;
		}
		else if (pe > ps + 2 && XP_STRNICMP(ps, "0b", 2) == 0)
		{
			ps += 2;
			literal_fmt = FMT_BIN;
		}
		else if (*(pe - 1) == 'b' || *(pe - 1) == 'B')
		{
			// check whether all value from ps to pe-1, all are 0 and 1
			auto pc = ps;
			for (; pc < pe - 1; pc++)
				if (*pc != '0' && *pc != '1')
					break;

			if (pc >= pe - 1)
			{
				pe--;
				literal_fmt = FMT_BIN;
			}
		}
		else if (*ps == '0' || *ps == 'o' || *ps == 'O')
		{
			ps++;
			literal_fmt = FMT_OCT;
		}
		else if (*(pe - 1) == 'o' || *(pe - 1) == 'O')
		{
			pe--;
			literal_fmt = FMT_OCT;
		}

		if (literal_fmt == FMT_AUTO)
		{
			// Check sign
			if (*ps == '-')
			{
				bNegative = true;
				ps++;
			}

			// still not decided
			auto pc = ps;
			for (; pc < pe; pc++)
			{
				if ((*pc >= 'a' && *pc <= 'f') || (*pc >= 'A' && *pc <= 'F'))
					literal_fmt = FMT_HEX;
				else if (*pc >= '0' && *pc <= '9')
				{
					if (literal_fmt == FMT_AUTO)
						literal_fmt = FMT_DEC;
				}
				else
					return false;	// It is not an valid value
			}
		}
	}
	else if (literal_fmt == FMT_HEX)
	{
		if (pe > ps + 2 && XP_STRNICMP(ps, "0x", 2) == 0)	// hex value
			ps += 2;
		else if (*(pe - 1) == 'h' || *(pe - 1) == 'H')
			pe--;
	}
	else if (literal_fmt == FMT_DEC)
	{
		if (*ps == '-')
		{
			bNegative = true;
			ps++;
		}
	}
	else if (literal_fmt == FMT_OCT)
	{
		if (*ps == '0' || *ps == 'o' || *ps == 'O')
			ps++;
		else if (*(pe - 1) == 'o' || *(pe - 1) == 'O')
			pe--;
	}
	else if (literal_fmt == FMT_BIN)
	{
		if (pe > ps + 2 && XP_STRNICMP(ps, "0b", 2) == 0)	// binary value
			ps += 2;
		else if (*(pe - 1) == 'b' || *(pe - 1) == 'B')
			pe--;
	}

	if (literal_fmt != FMT_HEX && literal_fmt != FMT_DEC && literal_fmt != FMT_OCT && literal_fmt != FMT_BIN)
		return false;

	while (ps < pe)
	{
		if (literal_fmt == FMT_HEX)	// hex value
		{
			if (*ps >= '0' && *ps <= '9')
				ret_val = (ret_val << 4) | (*ps - '0');
			else if (*ps >= 'a' && *ps <= 'f')
				ret_val = (ret_val << 4) | (*ps - 'a' + 10);
			else if (*ps >= 'A' && *ps <= 'F')
				ret_val = (ret_val << 4) | (*ps - 'A' + 10);
			else
				return false;
		}
		else if (literal_fmt == FMT_OCT)	// octal value
		{
			if (*ps >= '0' && *ps <= '7')
				ret_val = (ret_val << 3) | (*ps - '0');
			else
				return false;
		}
		else if (literal_fmt == FMT_DEC)		// decimal value
		{
			if (*ps >= '0' && *ps <= '9')
				ret_val = (ret_val * 10) + (*ps - '0');
			else
				return false;
		}
		else if (literal_fmt == FMT_BIN)
		{
			if (*ps >= '0' && *ps <= '1')
				ret_val = (ret_val << 1) | (*ps - '0');
			else
				return false;
		}
		else
			return false;

		ps++;
	}

	if (bNegative)
		ret_val = -1LL * ret_val;

	return true;
}

inline void PrintBuffer(uint8_t* buf, int buf_size, const char* cszLeadingStr)
{
	int ccWritten = -1;
	size_t ccBufLen = 80 * 2 + ((size_t)buf_size + 15) / 16 * 80;

	char* szBuffer = new char[ccBufLen + 1];
	char* szWriteBuf = szBuffer;

	printf("%s      00  01  02  03  04  05  06  07    08  09  0A  0B  0C  0D  0E  0F\n", cszLeadingStr);
	printf("%s      ----------------------------------------------------------------\n", cszLeadingStr);

	for (int idx = 0; idx < buf_size; idx++)
	{
		if (idx % 16 == 0)
		{
			ccWritten = XP_STPRINTF_S(szWriteBuf, ccBufLen, "%s %04X ", cszLeadingStr, idx); assert(ccWritten > 0);
			ccBufLen -= ccWritten; szWriteBuf += ccWritten; if (ccBufLen < 0)break;
		}

		ccWritten = XP_STPRINTF_S(szWriteBuf, ccBufLen, "%02X  ", buf[idx]); assert(ccWritten > 0);
		ccBufLen -= ccWritten; szWriteBuf += ccWritten; if (ccBufLen < 0)break;

		if ((idx + 1) % 8 == 0)
		{
			ccWritten = XP_STPRINTF_S(szWriteBuf, ccBufLen, "  "); assert(ccWritten > 0);
			ccBufLen -= ccWritten; szWriteBuf += ccWritten; if (ccBufLen < 0)break;
		}

		if ((idx + 1) % 16 == 0)
		{
			ccWritten = XP_STPRINTF_S(szWriteBuf, ccBufLen, "\n"); assert(ccWritten > 0);
			ccBufLen -= ccWritten; szWriteBuf += ccWritten; if (ccBufLen < 0)break;
		}
	}
	printf("%s\n", szBuffer);

	delete[] szBuffer;
}


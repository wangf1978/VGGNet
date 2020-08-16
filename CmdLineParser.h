#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "util.h"

enum NN_CMD
{
	NN_CMD_UNKNOWN = -1,
	NN_CMD_HELP = 0,
	NN_CMD_STATE,
	NN_CMD_TRAIN,
	NN_CMD_VERIFY,
	NN_CMD_CLASSIFY,
	NN_CMD_TEST,
};

enum NN_TYPE
{
	NN_TYPE_UNKNOWN = -1,
	NN_TYPE_LENET = 0,
	NN_TYPE_VGGA,
	NN_TYPE_VGGA_LRN,
	NN_TYPE_VGGB,
	NN_TYPE_VGGC,
	NN_TYPE_VGGD,
	NN_TYPE_VGGE,
};

struct CaseInsensitiveComparator
{
	bool operator()(const std::string& a, const std::string& b) const noexcept
	{
		return XP_STRICMP(a.c_str(), b.c_str()) < 0;
	}
};

enum OPTION_DATA_TYPE
{
	ODT_EMPTY = 0,
	ODT_BOOLEAN,
	ODT_INTEGER,
	ODT_FLOAT,
	ODT_STRING,
	ODT_OBJTYPE,
	ODT_MARKTYPE,
	ODT_TIMEUNIT,
	ODT_PRINTFMT,
	ODT_SHOWREF,
	ODT_LIST
};

struct COMMAND_OPTION
{
	const char*			option_name;
	const char*			short_tag;
	OPTION_DATA_TYPE	data_type;
	const char*			default_value_str;		// if there is no specified value, use this value
	void*				value_ref;
	bool				multiple;				// support multiple flags, and force to vector
	bool				switcher;				// the current flag is only a switcher, don't carry any parameter
};

class CmdLineParser
{
public:
	static bool ProcessCommandLineArgs(int argc, const char* argv[]);
	static CmdLineParser& GetCmdLineParser();
	~CmdLineParser();

	void Print();

	int32_t				verbose;					// 0: no verbose log, otherwise output the verbose log
	bool				silence;					// true: no prompt; false: my prompt to confirm or input something

	NN_CMD				cmd = NN_CMD_UNKNOWN;
	NN_TYPE				nn_type = NN_TYPE_VGGD;

	std::string			image_set_root_path;
	std::string			train_net_state_path;
	std::string			image_path;

	int32_t				batchsize = 1;				// the batch size to send the network, the default value 1
	int32_t				epochnum = 1;				// the epoch num to train the network, the default value 1
	float				learningrate = -1;			// the learning rate
	bool				enable_batch_norm = false;
	int32_t				num_classes = 1000;
	bool				use_32x32_input = false;
	int32_t				showloss_per_num_of_batches = 1;
	bool				clean_pretrain_net = false;

	std::vector<int>	cmd_args;

	int					argc = 0;
	const char**		argv = NULL;

protected:
	CmdLineParser();
	static CmdLineParser 
						m_cmdLineParser;
	static const std::map<std::string, NN_CMD, CaseInsensitiveComparator>
						mapCmds;
	static const std::map<std::string, NN_TYPE, CaseInsensitiveComparator>
						mapNNTypes;
	static COMMAND_OPTION
						options[];
	static COMMAND_OPTION
						cmd_flags[];

protected:
	void				parse_options(std::vector<int>& args, COMMAND_OPTION* options_table, size_t table_size, std::vector<int>& unparsed_arg_indexes);
	void				parse_cmdargs(std::vector<int32_t>& args);
};


#include "CmdLineParser.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

std::vector<std::string> split(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

const std::map<std::string, NN_CMD, CaseInsensitiveComparator> CmdLineParser::mapCmds =
{
	{"help",		NN_CMD_HELP},
	{"state",		NN_CMD_STATE},
	{"train",		NN_CMD_TRAIN},
	{"verify",		NN_CMD_VERIFY},
	{"classify",	NN_CMD_CLASSIFY},
	{"test",		NN_CMD_TEST}
};

const std::map<std::string, NN_TYPE, CaseInsensitiveComparator> CmdLineParser::mapNNTypes =
{
	{"LENET",		NN_TYPE_LENET},
	{"VGGA",		NN_TYPE_VGGA},
	{"VGGA_LRN",	NN_TYPE_VGGA_LRN},
	{"VGGB",		NN_TYPE_VGGB},
	{"VGGC",		NN_TYPE_VGGC},
	{"VGGD",		NN_TYPE_VGGD},
	{"VGGE",		NN_TYPE_VGGE},
};

COMMAND_OPTION CmdLineParser::options[] = {
	{"verbose",			"v",		ODT_INTEGER,		"1",		&CmdLineParser::m_cmdLineParser.verbose,				false,	false},
	{"quiet",			"y",		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.silence,				false,	false},
};

COMMAND_OPTION CmdLineParser::cmd_flags[] = {
	{"type",			"t",		ODT_INTEGER,		"5",		&CmdLineParser::m_cmdLineParser.nn_type,				false,	false},
	{"batchsize",		"b",		ODT_INTEGER,		"1",		&CmdLineParser::m_cmdLineParser.batchsize,				false,	false},
	{"epochnum",		"e",		ODT_INTEGER,		"1",		&CmdLineParser::m_cmdLineParser.epochnum,				false,	false},
	{"learningrate",	"l",		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.learningrate,			false,	false},
	{"lr",				"l",		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.learningrate,			false,	false},
	{"batchnorm",		NULL,		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.enable_batch_norm,		false,	true},
	{"bn",				NULL,		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.enable_batch_norm,		false,	true},
	{"numclass",		"n",		ODT_INTEGER,		NULL,		&CmdLineParser::m_cmdLineParser.num_classes,			false,	false},
	{"smallsize",		"s",		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.use_32x32_input,		false,	true},
	{"showloss",		NULL,		ODT_INTEGER,		NULL,		&CmdLineParser::m_cmdLineParser.showloss_per_num_of_batches,
																															false,	false},
	{"clean",			NULL,		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.clean_pretrain_net,		false,	true},
};

CmdLineParser CmdLineParser::m_cmdLineParser;

CmdLineParser& CmdLineParser::GetCmdLineParser()
{
	return m_cmdLineParser;
}

CmdLineParser::CmdLineParser()
{
}

CmdLineParser::~CmdLineParser()
{
}

void CmdLineParser::parse_options(std::vector<int>& args, COMMAND_OPTION* options_table, size_t table_size, std::vector<int>& unparsed_arg_indexes)
{
	int current_option_idx = -1;
	std::vector<std::string> parse_errors;
	auto iter = args.cbegin();
	while (iter != args.cend())
	{
		// check long pattern
		int cur_arg_idx = *iter++;
		const char* szItem = argv[cur_arg_idx];
		if (szItem[0] == '-' && strlen(szItem) > 1)
		{
			current_option_idx = -1;
			if (szItem[1] == '-')
			{
				// long format
				for (size_t i = 0; i < table_size; i++)
					if (XP_STRICMP(szItem + 2, options_table[i].option_name) == 0) {
						current_option_idx = (int)i;
						break;
					}
			}
			else if (szItem[1] != '\0' && szItem[2] == '\0')
			{
				for (size_t i = 0; i < table_size; i++)
					if (options_table[i].short_tag != NULL && strchr(options_table[i].short_tag, szItem[1]) != NULL) {
						current_option_idx = (int)i;
						break;
					}
			}

			if (current_option_idx < 0)
			{
				std::string err_msg = "Unrecognized option '";
				err_msg.append(szItem);
				err_msg.append("'");
				parse_errors.push_back(err_msg);
			}
			else
			{
				switch (options_table[current_option_idx].data_type)
				{
				case ODT_BOOLEAN:
					*((bool*)options_table[current_option_idx].value_ref) = options_table[current_option_idx].default_value_str != NULL &&
						XP_STRICMP(options_table[current_option_idx].default_value_str, "true") == 0 ? true : false;
					break;
				case ODT_INTEGER:
				{
					int64_t u64Val = 0;
					if (options_table[current_option_idx].default_value_str != NULL)
						ConvertToInt((char*)options_table[current_option_idx].default_value_str,
						(char*)options_table[current_option_idx].default_value_str + strlen(options_table[current_option_idx].default_value_str),
							u64Val);

					*((int32_t*)options_table[current_option_idx].value_ref) = (int32_t)u64Val;
					break;
				}
				case ODT_FLOAT:
				{
					double flVal = NAN;
					if (options_table[current_option_idx].default_value_str != NULL)
						flVal = atof(options_table[current_option_idx].default_value_str);
					*((float*)options_table[current_option_idx].value_ref) = (float)flVal;
					break;
				}
				case ODT_STRING:
				{
					std::string *strVal = (std::string*)options_table[current_option_idx].value_ref;
					if (strVal && options_table[current_option_idx].default_value_str != NULL)
						strVal->assign(options_table[current_option_idx].default_value_str);
					break;
				}
				case ODT_LIST:
					// Not implemented yet
					break;
				}
			}

			if (options_table[current_option_idx].switcher)
				current_option_idx = -1;

			continue;
		}

		if (current_option_idx >= 0)
		{
			switch (options_table[current_option_idx].data_type)
			{
			case ODT_BOOLEAN:
				*((bool*)options_table[current_option_idx].value_ref) = XP_STRICMP(szItem, "true") == 0 ? true : false;
				current_option_idx = -1;	// already consume the parameter
				break;
			case ODT_INTEGER:
			{
				int64_t u64Val = 0;
				ConvertToInt((char*)szItem, (char*)szItem + strlen(szItem), u64Val);
				*((int32_t*)options_table[current_option_idx].value_ref) = (int32_t)u64Val;
				current_option_idx = -1;	// already consume the parameter
				break;
			}
			case ODT_FLOAT:
			{
				double flVal = atof(szItem);
				*((float*)options_table[current_option_idx].value_ref) = (float)flVal;
				current_option_idx = -1;	// already consume the parameter
				break;
			}
			case ODT_STRING:
			{
				std::string *strVal = (std::string*)options_table[current_option_idx].value_ref;
				strVal->assign(szItem);
				current_option_idx = -1;	// already consume the parameter
				break;
			}
			case ODT_LIST:
				// Not implemented yet
				break;
			}
		}
		else
		{
			unparsed_arg_indexes.push_back(cur_arg_idx);
		}
	}
}

void CmdLineParser::parse_cmdargs(std::vector<int32_t>& args)
{
	std::vector<std::string> parse_errors;

	if (args.empty())
		return;

	if (cmd == NN_CMD_HELP)
	{

	}
	else if (cmd == NN_CMD_TRAIN)
	{

	}
	else if (cmd == NN_CMD_VERIFY)
	{

	}
	else if (cmd == NN_CMD_CLASSIFY)
	{

	}
	else if (cmd == NN_CMD_TEST)
	{

	}
}

bool CmdLineParser::ProcessCommandLineArgs(int argc, const char* argv[])
{
	std::vector<int> cmdoptions;
	std::vector<int> cmdargs;
	std::vector<int>* active_args = &cmdoptions;
	std::vector<int> unparsed_arg_indexes;

	m_cmdLineParser.argc = argc;
	m_cmdLineParser.argv = argv;

	for (int i = 1; i < argc; i++)
	{
		auto iter = mapCmds.find(argv[i]);
		if (iter != mapCmds.cend())
		{
			// hit the command
			m_cmdLineParser.cmd = iter->second;
			active_args = &cmdargs;
			continue;
		}

		active_args->push_back(i);
	}

	unparsed_arg_indexes.clear();
	m_cmdLineParser.parse_options(cmdoptions, options, sizeof(options) / sizeof(options[0]), unparsed_arg_indexes);

	unparsed_arg_indexes.clear();
	m_cmdLineParser.parse_options(cmdargs, cmd_flags, sizeof(cmd_flags) / sizeof(cmd_flags[0]), unparsed_arg_indexes);

	if (unparsed_arg_indexes.size() > 0)
	{
		if (m_cmdLineParser.cmd == NN_CMD_TRAIN)
		{
			m_cmdLineParser.image_set_root_path = argv[unparsed_arg_indexes[0]];
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[1]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_VERIFY)
		{
			m_cmdLineParser.image_set_root_path = argv[unparsed_arg_indexes[0]];
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[1]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_CLASSIFY)
		{
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[0]];
			m_cmdLineParser.image_path = argv[unparsed_arg_indexes[1]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_STATE)
		{
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[0]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_TEST)
		{

		}
	}

	if (m_cmdLineParser.verbose > 0)
	{
		printf("unparsed arguments:\n");
		for (auto u : unparsed_arg_indexes)
			printf("\t%s\n", argv[u]);
	}

	m_cmdLineParser.parse_cmdargs(unparsed_arg_indexes);

	return true;
}

void CmdLineParser::Print()
{
	printf("verbose: %d\n", verbose);
	printf("silence: %s\n", silence ? "yes" : "no");
	auto iterCmd = mapCmds.cbegin();
	for (; iterCmd != mapCmds.cend(); iterCmd++)
	{
		if (iterCmd->second == cmd)
		{
			printf("command: %s\n", iterCmd->first.c_str());
			break;
		}
	}

	if (iterCmd == mapCmds.cend())
		printf("command: Unknown\n");

	if (cmd == NN_CMD_TRAIN || cmd == NN_CMD_VERIFY)
		printf("image set path: %s\n", image_set_root_path.c_str());

	if (cmd == NN_CMD_TRAIN)
		printf("output train result: %s\n", train_net_state_path.c_str());
	else if (cmd == NN_CMD_VERIFY)
		printf("pre-trained net state: %s\n", train_net_state_path.c_str());

	if (cmd == NN_CMD_CLASSIFY)
	{
		printf("The file path of image to be clarified: %s\n", image_path.c_str());
		printf("The pre-trained net state: %s\n", train_net_state_path.c_str());
	}

	if (cmd == NN_CMD_TRAIN)
	{
		printf("batch size: %d\n", batchsize);
		printf("train epoch rounds: %d\n", epochnum);
		printf("learning rate: %f\n", learningrate);
		printf("enable batchnorm: %s\n", enable_batch_norm ? "yes" : "no");
		printf("num of output class: %d\n", num_classes);
		printf("the neutral network image input size: %s\n", use_32x32_input ? "32x32" : "224x224");
		printf("show loss per number of batches: %d\n", showloss_per_num_of_batches);
		printf("try to clean the previous train result: %s\n", clean_pretrain_net?"yes":"no");
	}
}
#include <Rcpp.h>

#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "compression/compress.h"
#include "gemma.h"    // Gemma
#include "util/app.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

struct gemma_interface {
  std::unique_ptr<gcpp::Gemma> model;
  std::unique_ptr<gcpp::LoaderArgs> loader;
  std::unique_ptr<gcpp::InferenceArgs> inference;
  std::unique_ptr<gcpp::AppArgs> app;
  std::unique_ptr<hwy::ThreadPool> inner_pool;
  std::unique_ptr<hwy::ThreadPool> pool;

  gcpp::AcceptFunc accept_token;

  int abs_pos;
  int current_pos;
  int prompt_size;
  std::mt19937 gen;

  std::vector<int> prompt_tokens;

  std::stringstream cur_response;
  std::vector<std::string> responses;
  std::vector<std::string> prompts;

  std::vector<std::string> arg_vec;

  std::vector<const char*> build_args(Rcpp::List const& args) {
    arg_vec.push_back("gemma");

    Rcpp::CharacterVector args_names = args.names();
    for (int i=0; i!=args.size(); ++i) {
      arg_vec.push_back("--" + std::string(args_names[i]));
      arg_vec.push_back(std::string(args[i]));
    }

    std::vector<const char*> argv;
    for (auto& arg : arg_vec) {
      argv.push_back(arg.c_str());
    }

    return argv;
  }

  gemma_interface(
    Rcpp::List args
  ) : abs_pos(0), current_pos(0), prompt_size(0)
  {
    auto argv = build_args(args);
    int argc = argv.size();

    loader    = std::unique_ptr<gcpp::LoaderArgs>(new gcpp::LoaderArgs(argc, (char**) &argv[0]));
    inference = std::unique_ptr<gcpp::InferenceArgs>(new gcpp::InferenceArgs(argc, (char**) &argv[0]));
    app       = std::unique_ptr<gcpp::AppArgs>(new gcpp::AppArgs(argc, (char**) &argv[0]));

    inner_pool = std::unique_ptr<hwy::ThreadPool>(new hwy::ThreadPool(0));
    pool       = std::unique_ptr<hwy::ThreadPool>(new hwy::ThreadPool(app->num_threads));

    // For many-core, pinning threads to cores helps.
    if (app->num_threads > 10) {
      gcpp::PinThreadToCore(app->num_threads - 1);  // Main thread
      pool->Run(
        0, pool->NumThreads(),
        [](uint64_t /*task*/, size_t thread) { gcpp::PinThreadToCore(thread); }
      );
    }

    model = std::unique_ptr<gcpp::Gemma>(new gcpp::Gemma(*loader, *pool));

    accept_token = [](int) { return true; };

    if (inference->deterministic) {
      gen.seed(42);
    } else {
      std::random_device rd;
      gen.seed(rd());
    }
  }

  void print_config() {
    int verbosity = app->verbosity;

    loader->Print(verbosity);
    inference->Print(verbosity);
    app->Print(verbosity);

    if (verbosity >= 2) {
      time_t now = time(nullptr);
      char* dt = ctime(&now);  // NOLINT
      Rcpp::Rcout << "Date & Time                   : " << dt
                  << "Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize                    << std::endl
                  << "Hardware concurrency          : " << std::thread::hardware_concurrency()        << std::endl
                  << "Instruction set               : " << hwy::TargetName(hwy::DispatchedTarget())
                  << " (" << hwy::VectorBytes() * 8 << " bits)" << std::endl
                  << "Weight Type                   : " << gcpp::TypeName(gcpp::WeightT())            << std::endl
                  << "EmbedderInput Type            : " << gcpp::TypeName(gcpp::EmbedderInputT())     << std::endl;
    }
  }

  std::string prompt(std::string prompt_string) {

    // callback function invoked for each generated token.
    auto stream_token = [this](int token, float) {
      auto& tokenizer = this->model->Tokenizer();

      int& abs_pos = this->abs_pos;
      int& current_pos = this->current_pos;
      int prompt_size = this->prompt_size;

      ++abs_pos;
      ++current_pos;
      if (current_pos < prompt_size) {
        //Rcpp::Rcerr << "." << std::flush;
      } else if (token == gcpp::EOS_ID) {
        if (!inference->multiturn) {
          abs_pos = 0;
          if (inference->deterministic) {
            gen.seed(42);
          }
        }
      } else {
        std::string token_text;
        HWY_ASSERT(tokenizer.Decode(std::vector<int>{token}, &token_text).ok());

        // +1 since position is incremented above
        if (current_pos == prompt_size + 1) {
          // first token of response
          token_text.erase(0, token_text.find_first_not_of(" \t\n"));
        }
        // TODO(austinvhuang): is explicit space necessary?
        Rcpp::Rcout << token_text << std::flush;
        this->cur_response << token_text;
      }

      return true;
    };

    if (abs_pos > inference->max_tokens) {
      Rcpp::Rcout << "max_tokens (" << inference->max_tokens << ") exceeded.\n"
                  << "Use a larger value if desired using the --max_tokens command line flag.\n";
    }

    current_pos = 0;
    prompts.push_back(prompt_string);

    if (prompt_string == "%c" || prompt_string == "%C") {
      abs_pos = 0;
    }

    if (model->model_training == gcpp::ModelTraining::GEMMA_IT) {
      // For instruction-tuned models: add control tokens.
      prompt_string = "<start_of_turn>user\n" + prompt_string +
        "<end_of_turn>\n"+
        "<start_of_turn>model\n";
      if (abs_pos > 0) {
        // Prepend "<end_of_turn>" token if this is a multi-turn dialogue
        // continuation.
        prompt_string = "<end_of_turn>\n" + prompt_string;
      }
    }

    HWY_ASSERT(model->Tokenizer().Encode(prompt_string, &prompt_tokens).ok());

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt_tokens.insert(prompt_tokens.begin(), 2);
    }

    prompt_size = prompt_tokens.size();
    GenerateGemma(*model, *inference, prompt_tokens, abs_pos, *pool, *inner_pool, stream_token, accept_token, gen, 1);
    Rcpp::Rcout << std::endl << std::endl;

    std::string res = cur_response.str();
    cur_response.str("");
    cur_response.clear();
    responses.push_back(res);

    return res;
  }

  std::string last_raw_prompt() {
    std::string res;
    auto& tokenizer = model->Tokenizer();

    std::string token_text;
    HWY_ASSERT(tokenizer.Decode(prompt_tokens, &token_text).ok());

    return token_text;
  }

  void reset() {
    current_pos = 0;
    abs_pos = 0;
    responses.clear();
    prompts.clear();
  }

  Rcpp::List status() {
    Rcpp::List L = Rcpp::List::create(
      Rcpp::Named("current_pos") = current_pos,
      Rcpp::Named("abs_pos") = abs_pos,
      Rcpp::Named("prompts") = prompts,
      Rcpp::Named("responses") = responses
    );

    return L;
  }
};

RCPP_MODULE(mod_gemma) {
  using namespace Rcpp;

  class_<gemma_interface>("gemma_interface")
  .constructor<Rcpp::List>()
  .method("prompt", &gemma_interface::prompt)
  .method("last_raw_prompt", &gemma_interface::last_raw_prompt)
  .method("reset", &gemma_interface::reset)
  .method("status", &gemma_interface::status)
  .method("print_config", &gemma_interface::print_config)
  ;
}


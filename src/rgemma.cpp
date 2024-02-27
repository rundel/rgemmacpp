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


void ShowConfig(gcpp::LoaderArgs& loader, gcpp::InferenceArgs& inference, gcpp::AppArgs& app) {
  loader.Print(app.verbosity);
  inference.Print(app.verbosity);
  app.Print(app.verbosity);

  if (app.verbosity >= 2) {
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

Rcpp::Environment base = Rcpp::Environment("package:base");
Rcpp::Function readline = base["readline"];


struct gemma_interface {
  std::unique_ptr<gcpp::Gemma> model;
  //gcpp::Gemma model;

  gcpp::LoaderArgs loader;
  gcpp::InferenceArgs inference;
  gcpp::AppArgs app;

  hwy::ThreadPool inner_pool;
  hwy::ThreadPool pool;
  
  gcpp::AcceptFunc accept_token;

  int abs_pos;
  int current_pos;
  int prompt_size;
  std::mt19937 gen;

  std::stringstream cur_response;
  std::vector<std::string> responses;

  gemma_interface(
    int argc, char** argv
  ) : loader(argc, argv), inference(argc, argv), app(argc, argv),
      inner_pool(0), pool(app.num_threads), //model(loader, pool),
      abs_pos(0), current_pos(0), prompt_size(0)
  {
    // For many-core, pinning threads to cores helps.
    if (app.num_threads > 10) {
      gcpp::PinThreadToCore(app.num_threads - 1);  // Main thread
      pool.Run(
        0, pool.NumThreads(),
        [](uint64_t /*task*/, size_t thread) { gcpp::PinThreadToCore(thread); }
      );
    }

    model = std::unique_ptr<gcpp::Gemma>(new gcpp::Gemma(loader, pool));

    accept_token = [](int) { return true; };

    if (inference.deterministic) {
      gen.seed(42);
    } else {
      std::random_device rd;
      gen.seed(rd());
    }
  }

  void show_config() {
    loader.Print(app.verbosity);
    inference.Print(app.verbosity);
    app.Print(app.verbosity);

    if (app.verbosity >= 2) {
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
      if (token == gcpp::EOS_ID) {
        if (!inference.multiturn) {
          abs_pos = 0;
          if (inference.deterministic) {
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


    if (abs_pos > inference.max_tokens) {
      Rcpp::Rcout << "max_tokens (" << inference.max_tokens << ") exceeded.\n"
                  << "Use a larger value if desired using the --max_tokens command line flag.\n";
    }

    std::vector<int> prompt;
    current_pos = 0;

    //if (prompt_string == "%q" || prompt_string == "%Q") {
    //  return;
    //}

    //if (prompt_string == "%c" || prompt_string == "%C") {
    //  abs_pos = 0;
    //  continue;
    //}

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

    HWY_ASSERT(model->Tokenizer().Encode(prompt_string, &prompt).ok());

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt.insert(prompt.begin(), 2);
    }

    prompt_size = prompt.size();

    //Rcpp::Rcout << "\nPrompt string:\n" 
    //  << "-------\n" 
    //  << prompt_string << std::endl 
    //  << "-------\n" 
    //  << std::flush;
    
    GenerateGemma(*model, inference, prompt, abs_pos, pool, inner_pool, stream_token, accept_token, gen, 1);
    
    std::string res = cur_response.str();

    cur_response.str("");
    cur_response.clear();
    
    //prompt_string = 

    responses.push_back(res);

    return res;
  }
};

// [[Rcpp::export]]
std::vector<std::string> gemmacpp(Rcpp::List args, std::string prompt, int verbosity = 1) {
  std::vector<std::string> args_vec = {"gemma"};
  Rcpp::CharacterVector args_names = args.names();
  for (int i=0; i!=args.size(); ++i) {
    args_vec.push_back("--" + std::string(args_names[i]));
    args_vec.push_back(std::string(args[i]));
  }

  //for(auto& x : args_vec) {
  //  Rcpp::Rcout << "\"" << x << "\"" << std::endl;
  //}

  int argc = args_vec.size();
  std::vector<const char*> argv;
  for (auto& arg : args_vec) {
    argv.push_back(arg.c_str());
  }

  gemma_interface obj(argc, (char**) &argv[0]);

  obj.show_config();

  obj.prompt(prompt);

  obj.prompt(prompt);

  return obj.responses;
}




void ReplGemma(
  gcpp::Gemma& model, hwy::ThreadPool& pool,
  hwy::ThreadPool& inner_pool, const gcpp::InferenceArgs& args,
  int verbosity, const gcpp::AcceptFunc& accept_token
) {

  int abs_pos = 0;      // absolute token index over all turns
  int current_pos = 0;  // token index within the current turn
  int prompt_size{};

  std::mt19937 gen;
  if (args.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  // callback function invoked for each generated token.
  auto stream_token = [
    &abs_pos, &current_pos, &args, &gen, &prompt_size,
    tokenizer = &model.Tokenizer(), verbosity
  ](int token, float) {
    ++abs_pos;
    ++current_pos;
    if (current_pos < prompt_size) {
      Rcpp::Rcerr << "." << std::flush;
    } else if (token == gcpp::EOS_ID) {
      if (!args.multiturn) {
        abs_pos = 0;
        if (args.deterministic) {
          gen.seed(42);
        }
      }
      if (verbosity >= 2) {
        Rcpp::Rcout << "\n[ End ]" << std::endl;
      }
    } else {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      // +1 since position is incremented above
      if (current_pos == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
        if (verbosity >= 1) {
          Rcpp::Rcout << std::endl << std::endl;
        }
      }
      // TODO(austinvhuang): is explicit space necessary?
      Rcpp::Rcout << token_text << std::flush;
    }
    return true;
  };

  while (abs_pos < args.max_tokens) {
    std::string prompt_string;
    std::vector<int> prompt;
    current_pos = 0;

    if (verbosity >= 1) {
      Rcpp::Rcout << "> " << std::flush;
    }
    //std::getline(std::cin, prompt_string);
    prompt_string = Rcpp::as<std::string>(readline("> "));

    if (prompt_string == "%q" || prompt_string == "%Q") {
      return;
    }

    if (prompt_string == "%c" || prompt_string == "%C") {
      abs_pos = 0;
      continue;
    }

    if (model.model_training == gcpp::ModelTraining::GEMMA_IT) {
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

    HWY_ASSERT(model.Tokenizer().Encode(prompt_string, &prompt).ok());

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt.insert(prompt.begin(), 2);
    }

    prompt_size = prompt.size();

    Rcpp::Rcerr << std::endl << "[ Reading prompt ] " << std::flush;

    const double time_start = hwy::platform::Now();
    GenerateGemma(model, args, prompt, abs_pos, pool, inner_pool, stream_token, accept_token, gen, verbosity);
    const double time_end = hwy::platform::Now();
    const double tok_sec = current_pos / (time_end - time_start);
    if (verbosity >= 2) {
      Rcpp::Rcout << current_pos << " tokens (" << abs_pos << " total tokens)" << std::endl
                  << tok_sec << " tokens / sec" << std::endl;
    }
    Rcpp::Rcout << std::endl << std::endl;
  }
  Rcpp::Rcout << "max_tokens (" << args.max_tokens << ") exceeded."
              << " Use a larger value if desired using the --max_tokens command line flag.\n";
}


// [[Rcpp::export]]
void test(std::string tokenizer, std::string compressed_weights, std::string model) {
  std::vector<std::string> args = {
    "gemma",
    "--tokenizer", tokenizer,
    "--compressed_weights", compressed_weights,
    "--model", model
  };

  int argc = args.size();
  std::vector<const char*> argv;
  for (auto& arg : args) {
    argv.push_back(arg.c_str());
  }

  gcpp::LoaderArgs loader(argc, (char**) &argv[0]);
  gcpp::InferenceArgs inference(argc, (char**) &argv[0]);
  gcpp::AppArgs app(argc, (char**) &argv[0]);

  if (const char* error = loader.Validate()) {
    Rcpp::stop("Invalid args: " + std::string(error));
  }

  if (const char* error = inference.Validate()) {
    Rcpp::stop("Invalid args: " + std::string(error));
  }

  hwy::ThreadPool inner_pool(0);
  hwy::ThreadPool pool(app.num_threads);

  // For many-core, pinning threads to cores helps.
  if (app.num_threads > 10) {
    gcpp::PinThreadToCore(app.num_threads - 1);  // Main thread
    pool.Run(
      0, pool.NumThreads(),
      [](uint64_t /*task*/, size_t thread) { gcpp::PinThreadToCore(thread); }
    );
  }

  gcpp::Gemma gemma_model(loader, pool);

  ShowConfig(loader, inference, app);

  ReplGemma(
    gemma_model, pool, inner_pool, inference, app.verbosity,
    /*accept_token=*/[](int) { return true; }
  );
}



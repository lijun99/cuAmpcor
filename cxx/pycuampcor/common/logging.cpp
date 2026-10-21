/**
 * @file  logging.cpp
 * @brief Messages with a level set for the process
 */

#include "public.h"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

namespace {
std::atomic<int> currentLevel{static_cast<int>(LogLevel::Info)};
std::mutex outputMutex;
} // namespace

LogLevel logLevel()
{
    return static_cast<LogLevel>(currentLevel.load());
}

void setLogLevel(LogLevel level)
{
    currentLevel = static_cast<int>(level);
}

void logMessage(LogLevel level, const std::string &message)
{
    std::lock_guard<std::mutex> lock(outputMutex);
    std::FILE *out = (level == LogLevel::Warning) ? stderr : stdout;
    std::fputs(message.c_str(), out);
    std::fputc('\n', out);
    std::fflush(out);
}

} // namespace
//end of file

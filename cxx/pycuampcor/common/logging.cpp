/**
 * @file  logging.cpp
 * @brief Messages with a level set for the process
 */

#include "logging.h"

namespace pycuampcor {

namespace {
std::atomic<int> currentLevel{static_cast<int>(LogLevel::Info)}; ///< the level of the messages written
std::mutex outputMutex; ///< serialize the writing of the messages among threads
} // namespace

/**
 * The level of the messages written
 */
LogLevel logLevel()
{
    return static_cast<LogLevel>(currentLevel.load());
}

/**
 * Set the level of the messages written (for the process)
 */
void setLogLevel(LogLevel level)
{
    currentLevel = static_cast<int>(level);
}

/**
 * Write a message (a line): warnings to stderr, the others to stdout, flushed
 */
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

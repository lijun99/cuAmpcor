/**
 * @file  logging.h
 * @brief Messages (progress, warnings, debugging) with a level set for the process
 *
 * Messages up to the level set (default Info) are written: warnings to stderr, the others to
 * stdout. Writing is thread-safe; a message is only formatted if its level is enabled.
 */

#pragma once

#include "externals.h"

namespace pycuampcor {

/// level of a message, or of the messages written
enum class LogLevel : int {
    Quiet = 0,    ///< no messages
    Warning = 1,  ///< warnings
    Info = 2,     ///< progress (default)
    Debug = 3,    ///< debugging
};

/// the level of the messages written
LogLevel logLevel();
/// set the level of the messages written
void setLogLevel(LogLevel level);

/// write a message (a line)
void logMessage(LogLevel level, const std::string &message);

/// write a message made of the arguments (streamed), if its level is enabled
template <typename... Args>
void logAt(LogLevel level, const Args &...args)
{
    if (level == LogLevel::Quiet || static_cast<int>(level) > static_cast<int>(logLevel()))
        return;
    std::ostringstream message;
    (message << ... << args);
    logMessage(level, message.str());
}

/// write a warning made of the arguments (streamed), if warnings are enabled
template <typename... Args>
void logWarning(const Args &...args) { logAt(LogLevel::Warning, args...); }

/// write a progress message made of the arguments (streamed), if info messages are enabled
template <typename... Args>
void logInfo(const Args &...args) { logAt(LogLevel::Info, args...); }

/// write a debugging message made of the arguments (streamed), if debug messages are enabled
template <typename... Args>
void logDebug(const Args &...args) { logAt(LogLevel::Debug, args...); }

} // namespace
//end of file

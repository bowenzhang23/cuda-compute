
#pragma once

#include "cuda_runtime.h"
#include <cstdio>

class Timer
{
public:
    static Timer& Instance()
    {
        static Timer timer;
        return timer;
    }

public:
    void         Tick(const cudaStream_t stream = 0);
    inline float ElapsedTime() const { return m_elapsedTime; }
    inline void  ShowElapsedTime(const char* tag) const
    {
        fprintf(stdout, "<<< Timer >>> %s spent %.4f ms\n", tag, m_elapsedTime);
    }

private:
    Timer();
    Timer(Timer& other)  = delete;
    Timer(Timer&& other) = delete;

private:
    bool        m_isRecording;
    float       m_elapsedTime;
    cudaEvent_t m_begin;
    cudaEvent_t m_end;
};
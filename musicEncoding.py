import numpy as np
import torch
from typing import List, Tuple
import mido

#have to admit I shamelessly used Claude 3.7 to generate this code, didn't feel like learning how to use these packages was worth the time or fit the aims of the project

class PianoPerformanceRepresentation:
    """
    Event-based representation for piano performances as described in the Music Transformer paper.
    
    The vocabulary consists of:
    - 128 NOTE_ON events (one for each MIDI pitch)
    - 128 NOTE_OFF events (one for each MIDI pitch)
    - 100 TIME_SHIFT events (in 10ms increments from 10ms to 1s)
    - 32 SET_VELOCITY events (quantized MIDI velocities)
    """
    
    def __init__(self):
        # Define vocabulary indices
        self.NOTE_ON_START = 0
        self.NOTE_ON_END = 127
        self.NOTE_OFF_START = 128
        self.NOTE_OFF_END = 255
        self.TIME_SHIFT_START = 256
        self.TIME_SHIFT_END = 355  # 100 time shifts (10ms to 1000ms in 10ms increments)
        self.VELOCITY_START = 356
        self.VELOCITY_END = 387  # 32 velocity bins
        
        self.vocab_size = 388  # Total vocabulary size
        
        # Define event mappings
        self._create_event_mappings()
    
    def _create_event_mappings(self):
        """Create mappings between events and their indices in the vocabulary."""
        # NOTE_ON events (MIDI pitches 0-127)
        self.pitch_to_note_on = {
            pitch: self.NOTE_ON_START + pitch 
            for pitch in range(128)
        }
        self.note_on_to_pitch = {v: k for k, v in self.pitch_to_note_on.items()}
        
        # NOTE_OFF events (MIDI pitches 0-127)
        self.pitch_to_note_off = {
            pitch: self.NOTE_OFF_START + pitch 
            for pitch in range(128)
        }
        self.note_off_to_pitch = {v: k for k, v in self.pitch_to_note_off.items()}
        
        # TIME_SHIFT events (10ms to 1000ms in 10ms increments)
        self.duration_to_time_shift = {
            (i + 1) * 10: self.TIME_SHIFT_START + i  # 10ms, 20ms, ..., 1000ms
            for i in range(100)
        }
        self.time_shift_to_duration = {v: k for k, v in self.duration_to_time_shift.items()}
        
        # SET_VELOCITY events (quantized into 32 bins)
        self.velocity_to_event = {
            int(i * 128 / 32) + min(i, 1): self.VELOCITY_START + i  # Adjustment to handle edge cases
            for i in range(32)
        }
        self.event_to_velocity = {v: k for k, v in self.velocity_to_event.items()}
    
    def encode_midi(self, midi_file: str) -> List[int]:
        """
        Encode a MIDI file into a sequence of event tokens.
        
        Args:
            midi_file: Path to the MIDI file
            
        Returns:
            List of event tokens
        """
        midi = mido.MidiFile(midi_file)
        events = []
        
        # Track active notes to handle sustain pedal
        active_notes = set()
        current_velocity = 0  # Default velocity
        sustain_pedal_active = False
        
        # Process MIDI messages
        current_time = 0
        for msg in mido.merge_tracks(midi.tracks):
            # Update current time
            if msg.time > 0:
                # Convert time from seconds to milliseconds
                time_ms = int(msg.time * 1000)
                
                # Split into 10ms TIME_SHIFT events
                while time_ms > 0:
                    shift = min(time_ms, 1000)  # Maximum 1000ms per event
                    time_ms -= shift
                    
                    # Find the closest available time shift
                    available_shifts = sorted(self.duration_to_time_shift.keys())
                    closest_shift = min(available_shifts, key=lambda x: abs(x - shift))
                    events.append(self.duration_to_time_shift[closest_shift])
            
            # Handle note events
            if msg.type == 'note_on' and msg.velocity > 0:
                # Add SET_VELOCITY event if velocity changed
                quantized_velocity = self._quantize_velocity(msg.velocity)
                if quantized_velocity != current_velocity:
                    current_velocity = quantized_velocity
                    events.append(self.velocity_to_event[quantized_velocity])
                
                # Add NOTE_ON event
                events.append(self.pitch_to_note_on[msg.note])
                active_notes.add(msg.note)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Add NOTE_OFF event
                events.append(self.pitch_to_note_off[msg.note])
                if msg.note in active_notes:
                    active_notes.remove(msg.note)
            
            # Handle sustain pedal
            elif msg.type == 'control_change' and msg.control == 64:  # Sustain pedal
                if msg.value >= 64 and not sustain_pedal_active:
                    sustain_pedal_active = True
                elif msg.value < 64 and sustain_pedal_active:
                    sustain_pedal_active = False
                    # Release all active notes that are held by the sustain pedal
                    for note in list(active_notes):
                        events.append(self.pitch_to_note_off[note])
                        active_notes.remove(note)
        
        return events
    
    def _quantize_velocity(self, velocity: int) -> int:
        """Quantize a MIDI velocity (0-127) to one of 32 bins."""
        bin_idx = min(31, velocity // 4)  # 128/32 = 4
        return int(bin_idx * 128 / 32) + min(bin_idx, 1)  # Adjustment to handle edge cases
    
    def decode_events(self, events: List[int]) -> List[Tuple]:
        """
        Decode a sequence of event tokens into a sequence of (type, value, time) tuples.
        
        Args:
            events: List of event tokens
            
        Returns:
            List of (type, value, time) tuples
        """
        result = []
        current_time = 0
        current_velocity = 0
        
        for event in events:
            if self.NOTE_ON_START <= event <= self.NOTE_ON_END:
                # NOTE_ON event
                pitch = self.note_on_to_pitch[event]
                result.append(('note_on', pitch, current_time, current_velocity))
                
            elif self.NOTE_OFF_START <= event <= self.NOTE_OFF_END:
                # NOTE_OFF event
                pitch = self.note_off_to_pitch[event]
                result.append(('note_off', pitch, current_time, 0))
                
            elif self.TIME_SHIFT_START <= event <= self.TIME_SHIFT_END:
                # TIME_SHIFT event
                duration = self.time_shift_to_duration[event]
                current_time += duration
                
            elif self.VELOCITY_START <= event <= self.VELOCITY_END:
                # SET_VELOCITY event
                current_velocity = self.event_to_velocity[event]
        
        return result
    
    def events_to_midi(self, events: List[int], output_file: str):
        """
        Convert a sequence of event tokens back to a MIDI file.
        
        Args:
            events: List of event tokens
            output_file: Path to save the MIDI file
        """
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        
        decoded = self.decode_events(events)
        last_time = 0
        
        for event_type, value, time, velocity in decoded:
            # Convert absolute time to delta time
            delta_time = time - last_time
            last_time = time
            
            # Create MIDI message
            if event_type == 'note_on':
                msg = mido.Message('note_on', note=value, velocity=velocity, time=delta_time)
                track.append(msg)
            elif event_type == 'note_off':
                msg = mido.Message('note_off', note=value, velocity=0, time=delta_time)
                track.append(msg)
        
        midi.save(output_file)
import { useEffect, useRef } from 'react';
import { RemoteParticipant, Room, RoomEvent } from 'livekit-client';

interface VoiceAgentRoomProps {
  url: string; // LiveKit server URL (wss://...)
  token: string; // LiveKit access token for the user
  agentIdentity?: string; // Optional: identity of the agent participant
}

export default function VoiceAgentRoom({ url, token, agentIdentity }: VoiceAgentRoomProps) {
  const roomRef = useRef<Room | null>(null);
  const agentAudioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    let room: Room;

    const join = async () => {
      room = new Room();
      await Promise.all([
        room.localParticipant.setMicrophoneEnabled(true),
        room.connect(url, token),
      ]);
      roomRef.current = room;

      // Listen for new tracks (agent's audio)
      room.on(RoomEvent.TrackSubscribed, (track, publication, participant: RemoteParticipant) => {
        if (
          track.kind === 'audio' &&
          agentAudioRef.current &&
          (!agentIdentity || participant.identity === agentIdentity)
        ) {
          // Attach agent's audio to the audio element
          const mediaStream = new MediaStream();
          mediaStream.addTrack(track.mediaStreamTrack);
          agentAudioRef.current.srcObject = mediaStream;
          agentAudioRef.current.play();
        }
      });
    };

    join();

    return () => {
      roomRef.current?.disconnect();
    };
  }, [url, token, agentIdentity]);

  return (
    <div className="mx-auto flex min-h-[40vh] w-full max-w-md flex-col items-center justify-center rounded-lg border border-zinc-200 bg-white p-4 shadow-lg sm:p-6 md:p-8 dark:border-zinc-800 dark:bg-zinc-900">
      <audio ref={agentAudioRef} autoPlay className="my-4 w-full max-w-xs" />
      {/* Responsive controls placeholder */}
      <div className="mt-2 flex w-full flex-col justify-center gap-2 sm:flex-row">
        {/* Example: Mute/Unmute and Leave buttons (not implemented) */}
        <button
          className="w-full rounded bg-blue-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-600 sm:w-auto"
          disabled
        >
          Mute/Unmute
        </button>
        <button
          className="w-full rounded bg-red-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-600 sm:w-auto"
          disabled
        >
          Leave
        </button>
      </div>
    </div>
  );
}
